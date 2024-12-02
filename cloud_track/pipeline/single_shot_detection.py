"""
Runs the tracker loop for a single prompt.
Does not feature re-initialization or object instance recovery.
"""

import os
import queue
import time
from typing import List

import cv2
import numpy as np
import PIL
import torch
from loguru import logger

from cloud_track.tracker_wrapper.co_tracker_wrapper import (
    FlexImage,
    VideoStreamer,
    visualizer,
)
from cloud_track.tracker_wrapper.opencv_wrapper import OpenCVWrapper
from cloud_track.utils import (
    get_best_box,
    get_best_mask,
    get_box_minimizing_cost_function,
    save_image,
)


def single_shot_detection_inner_cotracker(
    prompt,
    vLModel,
    stream: VideoStreamer,
    frontend_tracker,
    visualizers: List,
    loop_rate_timer,
    fm_timer,
    f_max=50,
    adaptive_fps=False,
    reinit_each=10,
    output_dir="/home/blei/cloud_track/results/",
):
    has_masks = False
    frame_number = 0
    iteration = 0
    last_iteration = False

    # frames to be processed before the stream. Used for failed frames.
    priority_video_buffer = queue.Queue()

    while True:

        try:
            if priority_video_buffer.empty():
                frame_np, success, gt = next(stream)

                if not success:
                    break
                # convert to PIL image
                frame_pil = PIL.Image.fromarray(frame_np)

                # check if frame is PIL image
                assert isinstance(frame_pil, PIL.Image.Image)
                flex_image = FlexImage(frame_np, gt=gt, frame_id=frame_number)
                frame_number += 1
            else:
                flex_image = priority_video_buffer.get()
                frame_np = flex_image.frame
                frame_pil = PIL.Image.fromarray(frame_np)

            print(
                f"Processing frame {flex_image.frame_id} in iteration {iteration}"
            )

            if not has_masks:
                # run grounded sam to find masks
                fm_timer.start()
                result_fm, masks, boxes, scores = vLModel.run_inference(
                    frame_pil, prompt
                )
                fm_timer.stop()

                # check, if mask is found
                if masks is None:
                    print("No masks found, Retrying ...")
                    continue
                else:
                    print("Masks found! Running coTracker ...")

                    # select the best mask only - for now ... change to mot later
                    masks = get_best_mask(masks, boxes, scores)

                    # image with stuff drawn on it
                    save_image(
                        result_fm,
                        "result_foundation_model.png",
                        basename=output_dir,
                    )  # future: use the wrapper_base method - obtain the underlying image from the wrapper

                    # save mask as image
                    masks_img = masks.squeeze().cpu().numpy() * 255
                    cv2.imwrite(
                        os.path.join(output_dir, "mask.png"), masks_img
                    )  # TODO: mit save_image speichern .. unten auch!
                    cv2.imwrite(
                        os.path.join(output_dir, "mask_source.png"), frame_np
                    )

                    # assert image is PIL and masks is torch.Tensor
                    assert isinstance(
                        masks, torch.Tensor
                    ), "masks must be torch.Tensor"

                    # convert result_fm to torch.Tensor
                    result_fm_torch = torch.from_numpy(
                        np.array(result_fm)
                    ).permute(2, 0, 1)

                    # add as keyframe to visualizer
                    [vis.add_keyframe(result_fm_torch) for vis in visualizers]

                    has_masks = True

                    flex_image.mask = masks

                    # set the mask at cotracker
                    reinit_request = frontend_tracker.run_inference(flex_image)

            else:
                # run coTracker
                reinit_request = frontend_tracker.run_inference(flex_image)

                # TODO: REMOVE THIS!
                if frame_number % reinit_each == 0:
                    frontend_tracker.trigger_reset(
                        reason="Scheduled re-initialization."
                    )

            # check if a reinit is requested
            if reinit_request is not None:
                logger.warning("Reinit request received.")
                logger.warning(str(reinit_request))

                # check if len of unprocessed frames is 0
                if not priority_video_buffer.empty():
                    raise Exception(
                        "Cannot add new frames to priority buffer since"
                        " it is not empty."
                    )

                for frame in reinit_request.unprocessed_frames:
                    priority_video_buffer.put(frame)
                has_masks = False

            # check, if cotracker is alive
            if not frontend_tracker.tracker.is_alive():
                raise Exception("CoTracker has died.")

            if adaptive_fps:
                coTracker_fps = frontend_tracker.get_loop_fps()
                if coTracker_fps is not None:
                    fps = coTracker_fps * 0.98  # as a margin of safety
                    print(f"Change to {fps} fps")
                    loop_rate_timer.set_framerate(fps)
                else:
                    print("Adaptive fps not initialized!")

            loop_rate_timer.regulate()

        except StopIteration:
            # end of sequence
            last_iteration = True
            break

        finally:
            if last_iteration:
                while frontend_tracker.is_bussy:
                    time.sleep(0.1)  # wait for cotracker to finish

            while frontend_tracker.out_queue.qsize() > 0:
                (
                    video_chunk,
                    pred_tracks,
                    pred_visibilities,
                    gt_batch,
                    frame_id_batch,
                ) = frontend_tracker.out_queue.get()

                # propagate boxes
                boxes_prop = frontend_tracker.propagate_box(pred_tracks, boxes)

                [
                    vis.add_batch(
                        video_chunk,
                        tracks=pred_tracks,
                        visibilities=pred_visibilities,
                        boxes=boxes_prop,
                        gt_batch=gt_batch,
                        frame_id_batch=frame_id_batch,
                    )
                    for vis in visualizers
                ]

            iteration += 1

        if frame_number == f_max:
            break


def single_shot_detection_inner_cv(
    prompt,
    vLModel,
    stream: VideoStreamer,
    tracker: OpenCVWrapper,
    visualizers: List,
    loop_rate_timer,
    fm_timer,
    f_max=50,
    adaptive_fps=False,
    reinit_each=-1,
    output_dir="/home/blei/cloud_track/results/",
    simulate_real_time=False,
    fps=30,
):
    has_masks = False
    frame_number = 0
    old_box = None
    bbox = None
    just_found_masks = False

    while True:
        try:
            if simulate_real_time and just_found_masks:
                # calculate the number of frames to skip
                n_to_skip = int((time.time() - fm_start_time) * fps)

                print(f"Skipping {n_to_skip} frames to simulate FM delay.")
                for i in range(n_to_skip):
                    frame_number += 1
                    frame_np, success, gt = next(stream)
                    bbox = None
                    for vis in visualizers:
                        vis.add_frame(
                            gt,
                            frame_number,
                            bbox,
                            img=frame_np,
                            is_fm_frame=False,
                        )
            else:
                frame_np, success, gt = next(stream)
        except StopIteration:
            break

        if not success:
            logger.warning("Image is None, thus leaving the loop.")
            break

        if frame_number % reinit_each == 0 and reinit_each > 0:
            has_masks = False
            old_box = bbox

        just_found_masks = False

        is_fm_frame = False
        if not has_masks:
            is_fm_frame = True
            fm_timer.start()
            fm_start_time = time.time()
            # convert image to pil
            frame_pil = PIL.Image.fromarray(frame_np)

            # run the foundation model to get the masks
            # raise NotImplementedError("Make the cathegory below configurable")
            result_fm, masks, boxes, scores = vLModel.run_inference(
                frame_pil, category="person", description=prompt
            )
            fm_timer.stop()
            fm_end_time = time.time()
            print(vLModel.get_stats())

            # convert result_fm to torch.Tensor
            result_fm_torch = torch.from_numpy(np.array(result_fm)).permute(
                2, 0, 1
            )
            for vis in visualizers:
                vis.add_keyframe(result_fm_torch)

            if masks is None:
                print("No masks found, Retrying ...")
                bbox = None
                for vis in visualizers:
                    vis.add_frame(
                        gt,
                        frame_number,
                        bbox,
                        frame_np,
                        is_fm_frame=is_fm_frame,
                    )
                continue

            # initialize the tracker with the best box
            print("Masks found! Running cvTracker ...")

            # get the box with the highest confidence
            if old_box is None:
                bbox = get_best_box(boxes, scores)
                bbox = bbox.squeeze()
            else:
                bbox = get_box_minimizing_cost_function(boxes, scores, old_box)

            scale_factor = 1
            frame_np_half = cv2.resize(
                frame_np,
                (
                    frame_np.shape[1] // scale_factor,
                    frame_np.shape[0] // scale_factor,
                ),
            )
            bbox_half = bbox / scale_factor

            tracker.init(frame_np_half, bbox_half)
            has_masks = True
            just_found_masks = True

        else:
            loop_rate_timer.start()
            # half resolution of the frame
            frame_np_half = cv2.resize(
                frame_np,
                (
                    frame_np.shape[1] // scale_factor,
                    frame_np.shape[0] // scale_factor,
                ),
            )

            # run cv tracker
            success, bbox_half = tracker.update(frame_np_half)
            bbox = torch.tensor(bbox_half) * scale_factor

            if not success:
                print("Object lost, retrying.")
                has_masks = False

            loop_rate_timer.stop()
            print(loop_rate_timer)

        for vis in visualizers:
            vis.add_frame(
                gt, frame_number, bbox, frame_np, is_fm_frame=is_fm_frame
            )

        # if just_found_masks:
        #    input("Press Enter to continue...")
        #    just_found_masks = False

        frame_number += 1

        if frame_number == f_max:
            break
