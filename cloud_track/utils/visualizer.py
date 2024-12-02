import json
import os

# check if python version is above 3.6
import sys
import time
from abc import abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch

if sys.version_info[0] >= 3 and sys.version_info[1] > 6:
    from torchvision.utils import draw_bounding_boxes, draw_keypoints


class output_wrapper:
    @abstractmethod
    def add_batch(self):
        pass

    @abstractmethod
    def add_keyframe(self):
        pass

    @abstractmethod
    def write_result(self, out_folder):
        pass


class gt_writer(output_wrapper):
    def __init__(self, fps, resolution=(1296, 720), no_window=False):
        self.res_gt = []

    def add_batch(
        self,
        video_buffer: torch.Tensor,
        tracks: torch.Tensor,
        visibilities: torch.Tensor,
        boxes: torch.Tensor = None,  # x_min, y_min, x_max, y_max
        gt_batch: np.ndarray = None,  # x_min, y_min, x_max, y_max
        frame_id_batch: torch.Tensor = None,
    ):

        for gt, frame_id, box in zip(gt_batch, frame_id_batch, boxes):
            self.add_frame(gt, frame_id, box)

    def add_frame(
        self,
        gt: np.ndarray,
        frame_id: np.int64,
        box: torch.Tensor,
        img: np.ndarray = None,
        is_fm_frame=False,
    ):
        # if gt is Torch, convert to ndarray
        if isinstance(gt, torch.Tensor):
            gt = gt.numpy()

        if isinstance(frame_id, int):
            frame_id = np.int64(frame_id)

        # assert types
        assert isinstance(gt, np.ndarray), "gt must be np.ndarray"
        assert isinstance(frame_id, np.int64), "frame_id must be np.int64"
        if box is not None:
            assert isinstance(box, torch.Tensor), "box must be torch.Tensor"

        box = list(map(float, box)) if box is not None else None
        frame_dict = {
            "frame_id": int(frame_id),
            "gt": list(map(float, gt)),
            "box": box,
            "timestamp": np.float64(time.time()),  # for fps calculation
            "is_fm_frame": is_fm_frame,
        }
        self.res_gt.append(frame_dict)

    def add_keyframe(self, keyframe: torch.Tensor):
        pass

    def write_result(self, out_folder):
        out_file = os.path.join(out_folder, "track.json")
        with open(out_file, "w") as f:
            json.dump(self.res_gt, f, indent=4)


class Visualizer(output_wrapper):
    def __init__(self, fps, resolution=(1296, 720), no_window=False):

        self.fps = fps
        self.res_video = []
        self.screen = None
        self.box_width = 3

        self.no_window = no_window

        if not (sys.version_info[0] >= 3 and sys.version_info[1] > 6):
            raise NotImplementedError(
                "Visual Rendering not supported on Python" " 3.6 and below."
            )

        if not self.no_window:
            self.window_name = "Flextrack"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("CoTracker Matches", *resolution)
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            cv2.waitKey(100)
            print("Visualizer initialized")

    def add_batch(
        self,
        video_buffer: torch.Tensor,
        tracks: torch.Tensor,
        visibilities: torch.Tensor,
        boxes: torch.Tensor = None,
        gt_batch: np.ndarray = None,
        frame_id_batch: torch.Tensor = None,
    ):
        assert video_buffer.shape[0] == tracks.shape[0]
        assert tracks.shape[0] == visibilities.shape[0]
        assert video_buffer.shape[0] == 1
        # assert torch.diff(torch.tensor(frame_id_batch)).max() <= 1 # max diff between frame ids is 1 -> Detect, if one is missed

        video_buffer = video_buffer.squeeze(1)
        tracks = tracks.squeeze(1)
        visibilities = visibilities.squeeze(1)

        if boxes is None:
            for img, track, vis in zip(
                video_buffer, tracks, visibilities
            ):  # this for loop does nothing afaik ...
                self.draw_frames(img, track, vis)
        else:
            for img, track, vis in zip(video_buffer, tracks, visibilities):
                self.draw_frames(img, track, vis, boxes, gt_batch)

    def draw_frames(
        self,
        video_chunk: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        boxes: torch.Tensor = None,
        gt_batch: np.ndarray = None,
    ):  #
        # draw the tracks on the image
        frame_no = 0
        for img, points, vis in zip(video_chunk, tracks, visibility):
            # we can add points for multiple instances. For now, we treat all
            # points as belonging to t#he same instance
            points = points.unsqueeze(0)

            # respect visibility
            visible_points = points[:, vis, :]
            invisible_points = points[:, ~vis, :]

            # convert image to uint8
            img = img.to(torch.uint8)

            # draw the points: green for visible, red for invisible
            img = draw_keypoints(
                img, visible_points, radius=2, colors=(0, 255, 0)
            )
            img = draw_keypoints(
                img, invisible_points, radius=2, colors=(255, 0, 0)
            )

            # draw the boxes on the image
            if boxes is not None:
                box = boxes[frame_no]
                img = draw_bounding_boxes(
                    img, box.unsqueeze(0), width=self.box_width
                )

            # draw the gt boxes on the image
            if gt_batch is not None:
                gt = gt_batch[frame_no]
                img = draw_bounding_boxes(
                    img,
                    torch.tensor(gt).unsqueeze(0),
                    colors=(0, 0, 255),
                    width=self.box_width,
                )

            # Convert the torch tensor to a numpy array
            numpy_image = img.numpy()

            # Convert the numpy array to a cv2 image
            cv2_image = np.transpose(numpy_image, (1, 2, 0))

            # Display the image using cv2
            if not self.no_window:
                cv2.imshow(self.window_name, cv2_image)
                # key = chr(cv2.waitKey(1) & 0xFF)  # noqa

            img = torch.from_numpy(cv2_image).permute(2, 0, 1)

            self.res_video.append(img)
            frame_no += 1

    def add_frame(
        self,
        gt: np.ndarray,
        frame_id: np.int64,
        box: torch.Tensor,
        img: np.ndarray = None,
        is_fm_frame=False,
    ):
        img = torch.from_numpy(img).permute(2, 0, 1)

        if gt is not None:
            # draw gt boxes on the image
            img = draw_bounding_boxes(
                img, gt.unsqueeze(0), colors=(0, 0, 255), width=self.box_width
            )

        if box is not None:
            # draw box on the image
            img = draw_bounding_boxes(
                img,
                box.unsqueeze(0),
                colors=(37, 255, 20),
                width=self.box_width,
            )

        # Convert the torch tensor to a numpy array
        numpy_image = img.numpy()

        # Convert the numpy array to a cv2 image
        cv2_image = np.transpose(numpy_image, (1, 2, 0))

        # Display the image using cv2
        if not self.no_window:
            cv2.imshow(self.window_name, cv2_image)
            key = chr(cv2.waitKey(1) & 0xFF)  # noqa
            # print("drawing ..")

        self.res_video.append(img)

    def add_keyframe(self, keyframe: torch.Tensor):

        # assert keyframe is torch.tensor
        assert isinstance(
            keyframe, torch.Tensor
        ), "keyframe must be torch.Tensor"

        keyframe = keyframe.squeeze(1)
        keyframe = keyframe.to(torch.uint8)

        # Display the image using cv2
        # np_frame = keyframe[0].permute(1, 2, 0).numpy()
        # if not self.no_window:
        #    cv2.imshow(self.window_name, np_frame)
        #    key = chr(cv2.waitKey(1) & 0xFF)  # noqa
        #     #print("drawing ..")

        # add the image to the video fps times
        for _ in range(self.fps * 3):  # wait 3 seconds
            self.res_video.append(keyframe)

    def write_result(self, out_folder, fps=None):
        # hide the windows
        if not self.no_window:
            cv2.destroyAllWindows()

        # save the video
        from torchvision.io import write_video

        fps = self.fps if fps is None else fps

        video_tensor = torch.stack(self.res_video)
        video_tensor = video_tensor.permute(0, 2, 3, 1)

        # change bgr to rgb
        video_tensor = video_tensor[:, :, :, [2, 1, 0]]

        out_file = os.path.join(out_folder, "track.mp4")
        print(f"Writing video to: {out_file}")

        write_video(out_file, video_tensor, self.fps, video_codec="vp9")
