import json
import time

import cv2
import numpy as np
from loguru import logger

from cloud_track.pipeline import CloudTrack
from cloud_track.rpc_communication.rpc_wrapper import RpcWrapper
from cloud_track.tracker_wrapper import OpenCVWrapper
from cloud_track.utils import VideoStreamer
from cloud_track.api_functions.live_demo import DashboardMaker


def run_video_demo(
    video_source,
    resolution,
    cathegory,
    description,
    frontend_tracker,
    frontend_tracker_threshold,
    backend_address,
    backend_port,
    output_file,
) -> None:

    logger.info("Starting CloudTrack Command Line Demo")
    logger.info(f"----------------------------------")
    logger.info(f"Connecting to backend at {backend_address}:{backend_port}")
    logger.info(
        f"Using frontend tracker {frontend_tracker} with threshold {frontend_tracker_threshold}"
    )
    logger.info(f"Using video source {video_source}")
    logger.info(f"Cathegory: '{cathegory}'")
    logger.info(f"Description: '{description}'")
    if output_file is not None:
        logger.info(f"Output file: '{output_file}'")
    logger.info(f"----------------------------------")

    backend = RpcWrapper(backend_address, backend_port)

    frontend_tracker = OpenCVWrapper(
        tracker_type=frontend_tracker,
        reinit_threshold=frontend_tracker_threshold,
    )

    cloud_track = CloudTrack(
        backend=backend, frontend_tracker=frontend_tracker
    )

    stream = VideoStreamer(
        source=video_source,
        resize=resolution,
    )
    
    window_name = "CloudTrack: Visualization"
    dashbard_maker = DashboardMaker(ui_theme="sar", window_name=window_name)
    
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
    )
    cv2.resizeWindow(window_name, int(resolution[0]*1.3), int(resolution[1]*1.3))

    results = []
    
    old_justification = "<None>"
    
    is_first_frame = True

    for idx, frame in enumerate(stream):
        
        if is_first_frame:
            # show a white frame for 1 second
            white_frame = np.ones_like(frame) * 255
            cv2.imshow(window_name, white_frame)
            cv2.waitKey(1)
            is_first_frame = False
            
        # get start time
        t_start = time.time()

        box, justification, score = cloud_track.forward(
            frame, category=cathegory, description=description
        )
        if score is not None:
            print(f"\rScore: {score:.5f}", end="")

        t_end = time.time()

        if justification is not None and justification != old_justification:
            logger.info(f"Found Object with Justification: {justification}")
            logger.info(f"Starting local tracking.")
            old_justification = justification

        if box is not None:
            box = [int(i) for i in box]
        else:
            logger.info("No Object Found. Trying with next frame ..")

        if output_file:
            results.append(
                {
                    "frame_idx": idx,
                    "box_xyxy": box,
                    "justification": justification,
                    "processing_time": t_end - t_start,
                }
            )
        
        # remder the dashbaord
        frame = dashbard_maker.render(
            hd_frame=frame,
            box=box,
            justification=justification,
            description=description,
            resize_height=1,
            resize_width=1,
            show_justification=True,
        )

        # convert frame to BGR and display
        cv2.imshow(
            window_name, np.array(frame, dtype=np.uint8)
        )
        cv2.waitKey(1)
    cv2.waitKey(10000)
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {output_file}")
    logger.info("Demo finished.")
