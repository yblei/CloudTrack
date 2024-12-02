from cloud_track.pipeline import CloudTrack
from cloud_track.utils import VideoStreamer
from cloud_track.tracker_wrapper import OpenCVWrapper
from cloud_track.rpc_communication.rpc_wrapper import RpcWrapper
from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import get_vlm_pipeline
import cv2
import numpy as np
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="demo")
def main(cfg: DictConfig) -> None:

    # SELECT THE VIDEO SOURCE
    video_source = "assets/apple.mp4"
    # video_source=0 # uncomment this line to use the webcam
    resolution = (640, 480)

    # SELECT THE PROMPTS (details in publication)
    cathegory = "an apple"  # this goes to the detector
    description = "Is this a red apple?"  # this goes to the VLM
    
    # SELECT THE FRONTEND TRACKER: Nano performs best in our tests
    frontend_tracker = "nano"
    frontend_tracker_threshold = 0.75

    # USE NETWORK BACKEND OR RUN BACKEND IN THE SAME PROCESS
    use_network_backend = (
        False  # set True to run the backend on the network
    )
    backend_address = "http://127.0.0.1"  # the backend ip (here: localhost)
    backend_port = 3000  # the backend port

    if use_network_backend:
        # In this case, vlm, detector and system prompt are set through the cli
        # -> see: python -m cloud_track backend --help
        backend = RpcWrapper(backend_address, backend_port)
    else:
        ############### DETECTOR CONFIGURATION ################
        # choose a detector
        #dino = "sam_lq"
        dino = "sam_hq"

        ############### VLM CONFIGURATION ################
        # Choose a VLM: Uncomment one of the following lines and the
        # appropriate  system prompt!

        # Llava configuration
        vlm = "llava-hf/llava-1.5-7b-hf"
        # vlm = "llava-hf/llava-1.5-13b-hf"
        # This system prompt is tuned for the llava models:
        system_prompt = "You should confirm if an object is in the image."

        # GPT configuration
        # vlm = "gpt-4o-mini"
        # vlm = "gpt-4o"
        # system_prompt = "You are an intelligent AI assistant that helps an "
        "object detection system to identify objects of different classes in"
        "images."  # This system prompt is tuned for the gpt-4 models

        backend = get_vlm_pipeline(
            vl_model_name=vlm,
            system_description=system_prompt,
            simulate_time_delay=False,
            detector_name=dino,
        )

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
        color_mode="rgb",
    )

    for frame in stream:
        box, justification = cloud_track.forward(
            frame, category=cathegory, description=description
        )
        if justification is not None:
            print("Justification: ", justification)

        if box is not None:
            box = [int(i) for i in box]
            cv2.rectangle(
                frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
            )

        # convert frame to BGR and display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("l", np.array(frame, dtype=np.uint8))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
