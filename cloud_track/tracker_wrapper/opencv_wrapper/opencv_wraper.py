"""
Wrapper around the OpenCV implementation of the tracker classes.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from .cv_models import (
    dasiamrpn_tracker_factory,
    goturn_tracker_factory,
    nano_tracker_factory,
    vit_tracker_factory,
)


class OpenCVWrapper:
    def __init__(
        self, tracker_type: str = "nano", reinit_threshold: float = 0.5
    ):
        self.tracker_type = tracker_type

        OPENCV_OBJECT_TRACKERS = (
            {  # falls das nicht geht -> installiere opencv-contrib-python
                "csrt": cv2.TrackerCSRT_create,
                "daSiamRpn": dasiamrpn_tracker_factory,
                "goturn": goturn_tracker_factory,
                "kcf": cv2.TrackerKCF_create,
                "mil": cv2.TrackerMIL_create,
                "nano": nano_tracker_factory,
                "vit": vit_tracker_factory,
            }
        )

        if self.tracker_type not in OPENCV_OBJECT_TRACKERS.keys():
            raise ValueError(
                f"Tracker type {self.tracker_type} not supported."
            )

        self.tracker = OPENCV_OBJECT_TRACKERS[self.tracker_type]()
        self.reinit_threshold = reinit_threshold

    def init(self, image: np.array, bbox: torch.Tensor):
        """
        Initialize the tracker with the first frame and the bounding box.

        image: np.array, the first frame
        bbox: torch.Tensor, the bounding box (cx, cy, w, h)
        """
        # torch to numpy
        bbox = bbox.numpy()
        bbox = bbox.astype(np.uint16)

        # convert xyxy to xmin ymin w h
        bbox = np.asarray(
            (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
            dtype=np.uint16,
        )

        # assert box is in frame
        # assert bbox[0] >= 0 and bbox[1] >= 0 and bbox[0] + bbox[2] < image.shape[1] and bbox[1] + bbox[3] < image.shape[0]
        logger.debug("Re-Initializing tracker.")
        self.tracker.init(image, bbox)

    def update(self, image):
        """
        Update the tracker with the next frame.

        returns:
            success: bool, whether the object was found again
            new_box: tuple, the new bounding box (x, y, w, h)
        """
        tracking_success, new_box = self.tracker.update(image)

        # convert xmin ymin to xyxy
        new_box = (
            int(new_box[0]),
            int(new_box[1]),
            int(new_box[0] + new_box[2]),
            int(new_box[1] + new_box[3]),
        )

        tracking_quality = self.__tracking_quality()
        success = tracking_success and tracking_quality

        return success, new_box

    def __tracking_quality(self):
        """
        Check, if the tracking has failed.

        returns:
            bool: True, if tracking is successfull, False otherwise.
        """
        supported_trackers = ["nano", "daSiamRpn", "vit"]
        if self.tracker_type not in supported_trackers:
            # raise NotImplementedError(f"Tracker type {self.tracker_type} does not support Tracking Score Calculation. ToDo: Find another way to evaluate tracking quality.")
            logger.warning(
                "Tracker type does not support tracking score calculation."
            )
            return True

        score = self.tracker.getTrackingScore()
        success = score > self.reinit_threshold

        logger.debug(f"Tracking score: {score}")
        if not success:
            logger.info(
                f"Tracking failed with score {score}. Reinitializing tracker."
            )

        return success

    def shutdown(self):
        """
        Shutdown the tracker.
        """
        pass

    def get_Timings(self):
        """
        Get the timings of the tracker.
        """
        pass


if __name__ == "__main__":

    tracker = OpenCVWrapper()
    print("OpenCVWrapper initialized.")
