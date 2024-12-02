# from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import DetectorVlmPipeline
import cv2
import numpy as np
import PIL

from cloud_track.tracker_wrapper import OpenCVWrapper
from cloud_track.utils import get_best_box, get_box_minimizing_cost_function
from cloud_track.utils.flow_control import PerformanceTimer


class CloudTrack:
    def __init__(
        self,
        backend,
        frontend_tracker: OpenCVWrapper,
    ):
        self.backend = backend
        self.frontend_tracker = frontend_tracker
        self.fm_timer = PerformanceTimer()

        # Initialization
        self.tracker_initialized = (
            False  # True, if we are tracking on the Frontend
        )
        self.box = None

    def reset(self):
        self.tracker_initialized = False
        self.box = None

    def forward(
        self, frame: np.ndarray, category: str, description: str = None
    ) -> tuple[tuple[int, int, int, int], str]:
        """Forward pass of the CloudTrack pipeline.

        Args:
            frame (np.ndarray): The frame in BGR format.
            category (str): The category of the object to track.
                This goes to the detector in the backend. (e.g. "an apple")
            description (str, optional): Verbal description of the object to
                track (e.g. "Is this a red apple?"). If no desctiption is
                provided, we generate one like f"Is this {category}?".


        Returns:
            tuple[tuple[int, int, int, int], str]: The bounding box of the tracked object in (xyxy format, dennoting top left and bottom right corner) and the justification for the decision.
        """
        # convret frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        justification = None
        if not self.tracker_initialized:
            bbox, justification = self.add_keyframe(
                frame, category, description
            )
            self.current_justification = justification
        else:
            bbox, success = self.track_frame(frame)

            if not success:
                self.reset()

        # convert bbox to numpy array
        if bbox is not None:
            bbox = np.array(bbox)

        return bbox, self.current_justification

    def add_keyframe(self, frame: np.ndarray, category: str, description: str):
        # convert frame to PIL
        image_pil = PIL.Image.fromarray(frame)

        with self.fm_timer:
            image_pil, masks, boxes_filt, scores, justifications = (
                self.backend.run_inference(image_pil, category, description)
            )

        selected_box = None
        justification = None

        if boxes_filt is not None:
            # get the best box
            if self.box is None:
                selected_box, idx = get_best_box(boxes_filt, scores)
            else:
                selected_box, idx = get_box_minimizing_cost_function(
                    boxes_filt, scores, self.box
                )

            selected_box = selected_box.squeeze()
            self.frontend_tracker.init(frame, selected_box)
            self.tracker_initialized = True
            self.box = selected_box
            justification = justifications[int(idx)]

        # convert bbox to numpy array
        if selected_box is not None:
            selected_box = np.array(selected_box)

        return selected_box, justification

    def track_frame(self, frame: np.ndarray):
        success, bbox = self.frontend_tracker.update(frame)

        return bbox, success
