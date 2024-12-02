import json
import os
import tempfile
from argparse import ArgumentParser
from os import path

import cv2
import numpy as np
import torch
from deva.ext.automatic_processor import (
    process_frame_automatic as process_frame_auto,
)
from deva.ext.automatic_sam import get_sam_model
from deva.ext.ext_eval_args import (
    add_auto_default_args,
    add_ext_eval_args,
    add_text_default_args,
)
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import (
    process_frame_with_text as process_frame_text,
)
from deva.inference.demo_utils import flush_buffer
from deva.inference.eval_args import (  # get_model_and_config,
    add_common_eval_args,
)
from deva.inference.inference_core import DEVAInferenceCore
from deva.model.network import DEVA
from tqdm import tqdm
from utils.deva_result_utils import ResultSaver

from .wrapper_base import WrapperBase


class DevaWrapper(WrapperBase):
    def __init__(self, cfg_in=None, deva_base_path="") -> None:

        np.random.seed(42)
        torch.autograd.set_grad_enabled(False)
        parser = ArgumentParser()
        add_common_eval_args(parser)
        add_ext_eval_args(parser)
        add_text_default_args(parser)
        deva_model, cfg, _ = get_model_and_config(
            parser, deva_base_path=deva_base_path
        )

        # default cfg values
        cfg["prompt"] = "bird.birds"
        cfg["enable_long_term_count_usage"] = True
        cfg["max_num_objects"] = 200
        cfg["size"] = 1080
        cfg["DINO_THRESHOLD"] = 0.3
        cfg["amp"] = True
        cfg["chunk_size"] = 8
        cfg["detection_every"] = 5
        cfg["max_missed_detection_count"] = 10
        cfg["sam_variant"] = "original"
        cfg["suppress_small_objects"] = False
        cfg["temporal_setting"] = "semionline"

        # overwrite from cfg_in
        if cfg_in is not None:
            for key in cfg_in:
                cfg[key] = cfg_in[key]

        deva_model = DEVAInferenceCore(deva_model, config=cfg)
        deva_model.next_voting_frame = cfg["num_voting_frames"] - 1
        deva_model.enabled_long_id()

        print("Configuration:", cfg)

        self.cfg = cfg
        self.deva_model = deva_model

        super().__init__()

    def predict(self, image):
        pass

    def print_results(self, boxes, scores, labels):
        pass

    def run_inference(
        self, video, video_out_path, print_results=True, mark_results=True
    ):
        """Runs inference on a video.

        Args:
            video (str): Path to video file.
            video_out_file (str): Path to output video file.
            print_results (bool, optional): Be more verbose. Defaults to True.
            mark_results (bool, optional): mark results on the image - not
                implemented. Defaults to True.
        """

        # obtain temporary directory
        result_saver = ResultSaver(
            output_root=os.path.dirname(video_out_path),
            video_name="temp/video.mp4",
            dataset="flextrack",
            object_manager=self.deva_model.object_manager,
        )
        result_saver.output_postfix = None
        writer_initizied = False

        gd_model, sam_model = get_grounding_dino_model(self.cfg, "cuda")

        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ti = 0
        # only an estimate
        with torch.cuda.amp.autocast(enabled=self.cfg["amp"]):
            with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret is True:
                        if not writer_initizied:
                            h, w = frame.shape[:2]
                            vid_folder = path.join(
                                tempfile.gettempdir(), "gradio-deva"
                            )
                            os.makedirs(vid_folder, exist_ok=True)
                            writer = cv2.VideoWriter(
                                video_out_path,
                                cv2.VideoWriter_fourcc(*"vp09"),
                                fps,
                                (w, h),
                            )
                            writer_initizied = True
                            result_saver.writer = writer

                        # construct frame name
                        frame_name = f"{ti:08d}"

                        process_frame_text(
                            self.deva_model,
                            gd_model,
                            sam_model,
                            frame_name,
                            result_saver,
                            ti,
                            image_np=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        )
                        ti += 1
                        pbar.update(1)
                    else:
                        break
            flush_buffer(self.deva_model, result_saver)

        # write the json from result saver
        json_data = result_saver.all_annotations
        json_path = os.path.join(os.path.dirname(video_out_path), "track.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        result_saver.end()
        writer.release()
        cap.release()
        self.deva_model.clear_buffer()
        return video_out_path


def get_model_and_config(parser: ArgumentParser, deva_base_path):
    args = parser.parse_args()
    config = vars(args)
    config["enable_long_term"] = not config["disable_long_term"]

    # change the checkpoint paths: append the deva_base_path to make them point to .saves folder
    for key, value in config.items():
        if type(value) is not str:
            continue
        if "./saves/" in value:
            config[key] = path.join(deva_base_path, value)

    # check, if exists
    if not path.exists(config["model"]):
        raise FileNotFoundError(
            f"Model not found: {config['model']}. Please run the download script first."
        )

    # Load our checkpoint
    network = DEVA(config).cuda().eval()
    if args.model is not None:
        model_weights = torch.load(args.model)
        network.load_weights(model_weights)
    else:
        print("No model loaded.")

    return network, config, args
