import json
import time

import cv2
import numpy as np
from loguru import logger

from cloud_track.pipeline import CloudTrack
from cloud_track.rpc_communication.rpc_wrapper import RpcWrapper
from cloud_track.tracker_wrapper import OpenCVWrapper
from cloud_track.utils import VideoStreamer


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

    results = []

    for idx, frame in enumerate(stream):
        # get start time
        t_start = time.time()

        box, justification = cloud_track.forward(
            frame, category=cathegory, description=description
        )

        t_end = time.time()

        if justification is not None:
            logger.info(f"Found Object with Justification: {justification}")
            logger.info(f"Starting local tracking.")

        if box is not None:
            box = [int(i) for i in box]
            cv2.rectangle(
                frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
            )
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

        # convert frame to BGR and display
        cv2.imshow(
            "CloudTrack: Visualization", np.array(frame, dtype=np.uint8)
        )
        cv2.waitKey(1)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
