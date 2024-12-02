from threading import Event, Lock, Thread

import cv2
import numpy as np
from loguru import logger

from cloud_track.pipeline import CloudTrack
from cloud_track.rpc_communication.rpc_wrapper import RpcWrapper
from cloud_track.tracker_wrapper import OpenCVWrapper
from cloud_track.utils import VideoStreamer

stop_event = Event()


def threaded_forward(
    cloud_track, frame, cathegory, description, box, justification
):
    logger.info("Querying backend ...")
    box_out, justification_out = cloud_track.forward(
        frame, category=cathegory, description=description
    )
    box = box_out
    justification = justification_out

    # logger.info("Backend response received.")

    return box, justification


class ThreadWithReturnValue(Thread):

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs={},
        Verbose=None,
    ):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
        self.result_lock = Lock()

    def run(self):
        if self._target is not None:
            _return = self._target(*self._args, **self._kwargs)
            self.result_lock.acquire()
            self._return = _return
            self.result_lock.release()

    def join(self, *args):
        Thread.join(self, *args)
        return self._return

    def get_results(self):
        self.result_lock.acquire()
        return self._return


def run_live_demo(
    video_source,
    tracker_resolution,
    cathegory,
    description,
    frontend_tracker,
    frontend_tracker_threshold,
    backend_address,
    backend_port,
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
    logger.info(f"----------------------------------")

    backend = RpcWrapper(backend_address, backend_port)

    frontend_tracker = OpenCVWrapper(
        tracker_type=frontend_tracker,
        reinit_threshold=frontend_tracker_threshold,
    )

    cloud_track = CloudTrack(
        backend=backend, frontend_tracker=frontend_tracker
    )

    # make viewer_resolution the same as tracker_resolution
    #viewer_resolution = (tracker_resolution[0]*2, tracker_resolution[1]*2)
    
    viewer_resolution = tracker_resolution

    stream = VideoStreamer(
        source=video_source,
        resize=viewer_resolution,
    )

    thread_running = False
    box = None
    justification = None
    just_initialized = False

    resize_width = viewer_resolution[0] / tracker_resolution[0]
    resize_height = viewer_resolution[1] / tracker_resolution[1]

    cv2.namedWindow("CloudTrack: Visualization")  # , cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("CloudTrack: Visualization",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    for hd_frame in stream:
        frame = cv2.resize(
            hd_frame, tracker_resolution, interpolation=cv2.INTER_AREA
        )
        if not cloud_track.tracker_initialized:
            if not thread_running:
                tmp_box = None
                tmp_justification = None
                t = ThreadWithReturnValue(
                    target=threaded_forward,
                    args=(
                        cloud_track,
                        frame,
                        cathegory,
                        description,
                        tmp_box,
                        tmp_justification,
                    ),
                )
                t.start()

                thread_running = True
                box = None
                justification = None

            if thread_running:
                if not t.is_alive():
                    thread_running = False
                    # box = t.get_results()[0]
                    # justification = t.get_results()[1]

        else:
            old_justification = justification
            box, justification = cloud_track.forward(
                frame, category=cathegory, description=description
            )
            just_initialized = (justification != old_justification) and (
                justification is not None
            )

        # print("Box: ", tmp_box)
        # print("Justification: ", tmp_justification)

        cv2.putText(
            hd_frame,
            description,
            (0, int(20 * resize_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 * resize_height,
            (255, 255, 255),
            1,
        )

        if just_initialized:
            logger.info(f"Found Object with Justification: {justification}")
            logger.info(f"Starting local tracking.")
            logger.info(
                "Klick the CloudTrack window and hold 'c' to change the prompts, 'r' to reset or 'q' to quit."
            )
            just_initialized = False

        if box is not None:
            box[0] = int(box[0] * resize_width)
            box[1] = int(box[1] * resize_height)
            box[2] = int(box[2] * resize_width)
            box[3] = int(box[3] * resize_height)
            

            # resize the box to the viewer resolution
            cv2.rectangle(
                hd_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
            )
            if justification is not None:
                frame_char_width = 60  # line break after 30 characters
                lines = [
                    justification[i : i + frame_char_width]
                    for i in range(0, len(justification), frame_char_width)
                ]
                for idx, line in enumerate(lines):
                    # put the description on the frame in black
                    cv2.putText(
                        hd_frame,
                        line,
                        (0, int(20 * (idx + 2) * resize_height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * resize_height,
                        (0, 255, 0),
                        1,
                    )
        else:
            pass

        # convert frame to BGR and display
        cv2.imshow(
            "CloudTrack: Visualization", np.array(hd_frame, dtype=np.uint8)
        )
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # reset on r
        if cv2.waitKey(1) & 0xFF == ord("r"):
            box = None
            justification = None
            cloud_track.reset()
        if cv2.waitKey(1) & 0xFF == ord("c"):
            box = None
            justification = None
            cloud_track.reset()

            old_cathegory = cathegory
            old_description = description

            cathegory = input("New cathegory (enter to reuse): ")
            description = input("New description (enter to reuse): ")

            if cathegory == "":
                cathegory = old_cathegory

            if description == "":
                description = old_description

            justification = None
