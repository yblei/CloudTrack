import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.vot_dataset import RefVotDataset
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cloud_track.utils.flow_control import LoopRateTimer, PerformanceTimer
from cloud_track.pipeline.single_shot_detection import (
    single_shot_detection_inner_cv,
)
from cloud_track.foundation_model_wrappers import GroundedSamWrapper

from cloud_track.foundation_model_wrappers.deva_wrapper import DevaWrapper
import deva
import json
from loguru import logger


@hydra.main(config_path="../conf", config_name="ref_vot_evaluation")
def main(cfg: DictConfig):
    """Run evaluation on VOT22 dataset.

    Args:
        cfg (DictConfig): The hydra config.
    """
    vot_root = cfg.dataset.vot_root

    ref_vot_dataset = RefVotDataset(vot_root)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    deva_base_path = os.path.dirname(os.path.dirname(deva.__file__))

    for sequence_query, sequence_dataset in ref_vot_dataset:
        print(sequence_query)

        # make directory
        sequence_dir = os.path.join(hydra_dir, sequence_dataset.sequence_name)
        os.makedirs(sequence_dir, exist_ok=True)

        # make cfg_in
        cfg_in = {}
        cfg_in["prompt"] = sequence_query

        model = DevaWrapper(deva_base_path=deva_base_path, cfg_in=cfg_in)

        # no_windows = cfg.visualizer.no_windows
        f_max = cfg.f_max
        reinit_each = cfg.reinit_each
        # resolution = cfg.source.resolution

        # print sequence name
        print(f"Processing: {sequence_dataset.sequence_name}")

        # assemble video_out_path
        video_out_path = os.path.join(sequence_dir, "output.mp4")

        generic_image_path = os.path.join(
            sequence_dataset.get_sequence_path(), "%08d.jpg"
        )
        model.run_inference(
            generic_image_path,
            video_out_path,
            print_results=True,
            mark_results=True,
        )

        # add the ground truth to the produced track.json
        track_path = os.path.join(sequence_dir, "track.json")
        add_ground_truth_to_track(track_path, sequence_dataset)


def add_ground_truth_to_track(track_path: str, sequence_dataset: RefVotDataset):
    """Add the ground truth to the track.json file.

    Args:
        track_path (str): The path to the track.json file.
        sequence_dataset (RefVotDataset): The dataset object.
    """
    with open(track_path, "r") as f:
        track = json.load(f)

    for i, (image, success, gt) in enumerate(sequence_dataset):
        try:
            assert (
                track[i]["frame_id"] == i
            ), "The frame_id in the track.json is not consistent with the frame number."
            track[i]["gt"] = gt.tolist()
        except IndexError:
            logger.warning(
                f"Index {i} not found in track.json. Seems like a frame was lost during analysis."
            )

    with open(track_path, "w") as f:
        json.dump(track, f, indent=4)

    assert check_key_consistency(
        track
    ), "The keys in the track.json are not consistent."


def check_key_consistency(track_json):
    """Check, if every element in track.json has the same keys."""
    keys = set(track_json[0].keys())
    for i in range(1, len(track_json)):
        if set(track_json[i].keys()) != keys:
            return False
    return True


if __name__ == "__main__":
    main()
