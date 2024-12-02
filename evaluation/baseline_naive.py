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
from cloud_track.tracker_wrapper.co_tracker_wrapper import (
    CoTrackerWrapper,
    VideoStreamer,
    visualizer,
)
import PIL
from tqdm import tqdm
import numpy as np
import torch

def baseline_pipeline(
    dataset,
    prompt: str,
    vLModel: GroundedSamWrapper = None,
    resolution: tuple = None,
    no_windows: bool = False,
    fps: int = 15,
    visualizer_fps: int = 15,  # fps of generated video
    interactive: bool = False,
    f_max: int = -1,
    output_dir="/home/blei/cloud_track/results/",
    reinit_each=10,
):
    visualizers = []
    visualizers.append(
        visualizer.gt_writer(no_window=no_windows, fps=visualizer_fps)
    )

    for frame_number, (frame_np, success, gt) in tqdm(enumerate(dataset)):
        if not success:
            break

        frame_pil = PIL.Image.fromarray(frame_np)

        result_fm, masks, boxes, scores = vLModel.run_inference(
            frame_pil, prompt
        )

        if len(scores) > 0:
            best_box = vLModel.get_best_box(boxes, scores)
        else:
            best_box = torch.Tensor([[0, 0, 0, 0]]) # empty box is treated as no detection

        # convert to numpy
        gt = np.array(gt)
        frame_number = np.int64(frame_number)
        best_box = best_box.squeeze()

        for vis in visualizers:
            vis.add_frame(gt, frame_number, best_box)
    
    for vis in visualizers:
        vis.write_result(output_dir)


@hydra.main(config_path="../conf", config_name="ref_vot_evaluation")
def main(cfg: DictConfig):
    """Run evaluation on VOT22 dataset.

    Args:
        cfg (DictConfig): The hydra config.
    """
    vot_root = cfg.dataset.vot_root

    ref_vot_dataset = RefVotDataset(vot_root)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for sequence_query, sequence_dataset in ref_vot_dataset:
        print(sequence_query)

        # make directory
        sequence_dir = os.path.join(hydra_dir, sequence_dataset.sequence_name)
        os.makedirs(sequence_dir, exist_ok=True)

        model = GroundedSamWrapper(use_sam_hq=True)

        # no_windows = cfg.visualizer.no_windows
        f_max = cfg.f_max
        reinit_each = cfg.reinit_each
        # resolution = cfg.source.resolution

        # print sequence name
        print(f"Processing: {sequence_dataset.sequence_name}")

        baseline_pipeline(
            dataset=sequence_dataset,
            prompt=sequence_query,
            vLModel=model,
            no_windows=True,
            fps=4,
            interactive=False,
            f_max=f_max,
            output_dir=sequence_dir,
            reinit_each=reinit_each,
        )


if __name__ == "__main__":
    main()
