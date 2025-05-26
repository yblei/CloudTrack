from threading import Event, Lock, Thread

import cv2
import numpy as np
from loguru import logger

from cloud_track.pipeline import CloudTrack
from cloud_track.rpc_communication.rpc_wrapper import RpcWrapper
from cloud_track.tracker_wrapper import OpenCVWrapper
from cloud_track.utils import VideoStreamer

TKINTER_AVAILABLE = True
try:
    import tkinter as tk
    from tkinter import simpledialog
except ImportError:
    TKINTER_AVAILABLE = False
    logger.warning(
        "tkinter is not available. The GUI for changing the prompts will not be available."
    )
    logger.warning(
        "Run \"sudo apt install python3-tk\" to install tkinter."
    )

stop_event = Event()


def threaded_forward(
    cloud_track, frame, cathegory, description, box, justification
):
    logger.info("Querying backend ...")
    box_out, justification_out, _ = cloud_track.forward(
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
    viewer_resolution,
    cathegory,
    description,
    frontend_tracker,
    frontend_tracker_threshold,
    backend_address,
    backend_port,
    ui_theme,
    fullscreen,
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
    logger.info(
    "Klick the CloudTrack window and hold 'c' to change the prompts, 'r' to reset or 'q' to quit."
)

    backend = RpcWrapper(backend_address, backend_port)


    frontend_tracker = OpenCVWrapper(
        tracker_type=frontend_tracker,
        reinit_threshold=frontend_tracker_threshold,
    )

    cloud_track = CloudTrack(
        backend=backend, frontend_tracker=frontend_tracker
    )

    tracker_resolution = (640, 480)

    # viewer_resolution = tracker_resolution

    stream = VideoStreamer(
        source=video_source,
        resize=viewer_resolution,
    )
    
    window_name = "CloudTrack: Visualization"
    dashboard_maker = DashboardMaker(ui_theme, window_name, fullscreen)

    thread_running = False
    box = None
    justification = None
    just_initialized = False

    resize_width = viewer_resolution[0] / tracker_resolution[0]
    resize_height = viewer_resolution[1] / tracker_resolution[1]

    # Create a named window with normal flags
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

    # wait for 500ms
    if fullscreen:
        cv2.waitKey(500)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    else:
        # make the window large
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
        )
        cv2.resizeWindow(window_name, int(viewer_resolution[0]*1.3), int(viewer_resolution[1]*1.3))

    for hd_frame in stream:       
        
        frame = cv2.resize(
            hd_frame, tracker_resolution, interpolation=cv2.INTER_AREA
        )
        score = None
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
            box, justification, score = cloud_track.forward(
                frame, category=cathegory, description=description
            )
            just_initialized = (justification != old_justification) and (
                justification is not None
            )
            
        if score is not None:
            # delete the last line in termin
            print(f"\rScore: {score:.5f}", end="")


        if just_initialized:
            logger.info(f"Found Object with Justification: {justification}")
            logger.info(f"Starting local tracking.")
            just_initialized = False

        hd_frame = dashboard_maker.render(
            hd_frame,
            box,
            justification,
            description,
            resize_width,
            resize_height,
        )

        # convert frame to BGR and display
        cv2.imshow(window_name, np.array(hd_frame, dtype=np.uint8))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        # reset on r
        if key == ord("r"):
            box = None
            justification = None
            cloud_track.reset()
        if key == ord("c"):
            box = None
            justification = None
            cloud_track.reset()

            old_cathegory = cathegory
            old_description = description
            
            # use tkinter to get input
            if TKINTER_AVAILABLE:
                cathegory, description = get_inputs_tkinter(description, cathegory)
            else:
                cathegory = input("New cathegory (enter to reuse): ")
                description = input("New description (enter to reuse): ")
                
            if cathegory == "":
                cathegory = old_cathegory
            if description == "":
                description = old_description
            logger.info(f"New cathegory: {cathegory}")
            logger.info(f"New description: {description}")

            justification = None
            

def get_inputs_tkinter(old_description, old_cathegory):
    def on_ok():
        nonlocal cat_input, desc_input
        result["category"] = cat_input.get()
        result["description"] = desc_input.get()
        dialog.destroy()

    def on_cancel():
        result["category"] = None
        result["description"] = None
        dialog.destroy()

    result = {}
    dialog = tk.Tk()
    dialog.title("Enter Details")

    # Layout
    tk.Label(dialog, text="New category:").pack(padx=10, pady=(10, 0))
    cat_input = tk.Entry(dialog, width=40)
    cat_input.insert(0, old_cathegory)
    cat_input.pack(padx=10, pady=(0, 10))

    tk.Label(dialog, text="New description:").pack(padx=10, pady=(0, 0))
    desc_input = tk.Entry(dialog, width=40)
    desc_input.insert(0, old_description)
    desc_input.pack(padx=10, pady=(0, 10))

    # Buttons
    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=(0, 10))
    tk.Button(button_frame, text="OK", width=10, command=on_ok).pack(side="left", padx=5)
    tk.Button(button_frame, text="Cancel", width=10, command=on_cancel).pack(side="right", padx=5)

    dialog.mainloop()
    return result["category"], result["description"]


class DashboardMaker:
    def __init__(self, ui_theme, window_name, fullscreen):
        # load assets/alert_sign.png
        alert_sign = cv2.imread("assets/alert_sign.png")

        # replace alpha channel with black
        alert_sign = cv2.cvtColor(alert_sign, cv2.COLOR_BGRA2BGR)
        
        self.alert_sign = alert_sign
        self.ui_theme = ui_theme
        self.window_name = window_name
        self.fullscreen = fullscreen
        
    def render(
        self,
        hd_frame,
        box,
        justification,
        description,
        resize_width,
        resize_height,
        show_justification=False,
    ):
        ui_theme = self.ui_theme
        window_name = self.window_name
        
        if ui_theme == "good":
            box_color = (0, 255, 0)
            text_color = box_color
            background_color_match = (0, 104, 0)
            background_color_default = (0, 0, 0)
            show_justification = True
        elif ui_theme == "sar":
            background_color_default = (0, 0, 0)
            box_color = (0, 0, 255)
            text_color = (255, 255, 255)
            background_color_match = (11, 31, 208)

        if box is not None:
            box = scale_box(box, resize_width, resize_height)
            background_color = background_color_match

            # resize the box to the viewer resolution
            cv2.rectangle(
                hd_frame, (box[0], box[1]), (box[2], box[3]), box_color, 2
            )
        else:
            background_color = background_color_default

        if self.fullscreen:
            w = 1600
            h = 1000
        else:
            _, _, w, h = cv2.getWindowImageRect(window_name)
    

        # if smaller than hd frame size return only the hd_frame
        if w < hd_frame.shape[1] or h < hd_frame.shape[0]:
            return hd_frame

        # this is a black background
        final_frame = np.ones((h, w, 3), dtype=np.uint8)

        # add the hd_frame to the final frame and center it
        x_offset = int(max((w - hd_frame.shape[1]) // 2, 0))
        y_offset = int(max((h - hd_frame.shape[0]) // 2, 0))

        # fill the background with the background color
        # and the color of the background
        final_frame[y_offset : y_offset + hd_frame.shape[0], :] = background_color

        final_frame[
            y_offset : y_offset + hd_frame.shape[0],
            x_offset : x_offset + hd_frame.shape[1],
        ] = hd_frame

        # if theme == "alert", add alert symbol to the top right corner of the window

        if ui_theme == "sar" and box is not None:

            # scale alert sign to 30% of original size
            w_sign, h_sign = self.alert_sign.shape[1], self.alert_sign.shape[0]

            alert_sign_y = int(10)

            sign_target_h = y_offset - 2 * alert_sign_y
            sign_scale_factor = sign_target_h / h_sign
            
            if sign_scale_factor > 0.05:
                # display alert sign if window is large enough
                alert_sign_resized = cv2.resize(
                    self.alert_sign,
                    (int(sign_scale_factor * w_sign), int(sign_scale_factor * h_sign)),
                    interpolation=cv2.INTER_AREA,
                )
                alert_sign_x = int(w - alert_sign_resized.shape[1] - 10)

                final_frame[
                    alert_sign_y : alert_sign_y + alert_sign_resized.shape[0],
                    alert_sign_x : alert_sign_x + alert_sign_resized.shape[1],
                ] = alert_sign_resized

        # to fix: Add alert symbol to the top right corner of the window
        # when theme == "alert" and match
        # Zeilenumbruch verbessern, zeichenlimit anheben
        # heartbeat bei match auf den hintergrund legen
        # Bild dynamisch anpassen (Rahmen sollte immer gleich groß bleiben)
        
        cv2.putText(
            final_frame,
            description,
            (x_offset, y_offset - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 * resize_height,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        
        if box is not None and justification is not None and show_justification:
            frame_char_width = 90  # line break after 30 characters
            lines = [
                justification[i : i + frame_char_width]
                for i in range(0, len(justification), frame_char_width)
            ]
            text_y_offset_base = y_offset + hd_frame.shape[0] + 5
            for idx, line in enumerate(lines):
                # put the description on the frame in black
                cv2.putText(
                    final_frame,
                    line,
                    (x_offset, int(20 * (idx+1) * resize_height + text_y_offset_base)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * resize_height,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        return final_frame


def scale_box(box, resize_width, resize_height):
    box_out = np.zeros(4, dtype=int)
    box_out[0] = int(box[0] * resize_width)
    box_out[1] = int(box[1] * resize_height)
    box_out[2] = int(box[2] * resize_width)
    box_out[3] = int(box[3] * resize_height)

    return box_out
