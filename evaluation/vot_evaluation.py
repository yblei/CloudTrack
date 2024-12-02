from utils.vot_dataset import RefVotDataset
import hydra
from omegaconf import DictConfig
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cloud_track.utils.flow_control import LoopRateTimer, PerformanceTimer
from cloud_track.pipeline.single_shot_detection import (
    single_shot_detection_inner_cv,
)
from cloud_track.foundation_model_wrappers import GroundedSamWrapper
from cloud_track.tracker_wrapper.co_tracker_wrapper import (
    CoTrackerWrapper,
    visualizer,
)


def vot_pipeline(
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
    """_summary_

    Args:
        dataset (str, optional): _description_. Defaults to 0.
        prompt (str): _description_.
        vLModel (GroundedSamWrapper, optional): _description_. Defaults to None.
        resolution (tuple, optional): Resizes to this resolution.
            Defaults to None.
        no_windows (bool, optional): _description_. Defaults to False.
        fps (int, optional): The initial fps of the tracker loop. Is changed
            adaptively, when using interactive mode. Defaults to 15.
        interactive (bool, optional): _description_. Defaults to False.
        f_max (int, optional): break after f_max frames.
            Defaults to -1 (No Limit).
    """
    # get actual resolution from stream. Only differs if resolution = None.
    resolution = dataset.get_resolution()

    coTracker = CoTrackerWrapper(
        frame_resolution=resolution
    )  # resolution important for grid sampling H, W

    try:
        visualizers = []
        visualizers.append(
            visualizer.Visualizer(no_window=no_windows, fps=visualizer_fps)
            )
        visualizers.append(
            visualizer.gt_writer(no_window=no_windows, fps=visualizer_fps)
        )

        loop_rate_timer = LoopRateTimer(framerate=fps)
        fm_timer = PerformanceTimer()

        single_shot_detection_inner_cv(
            prompt,
            vLModel,
            dataset,
            coTracker,
            visualizers,
            loop_rate_timer,
            fm_timer,
            f_max=f_max,
            adaptive_fps=True,
            reinit_each=reinit_each,
            output_dir=output_dir,
        )

        print(f"FM: {fm_timer}")
        print(f"PT (per batch): {coTracker.get_timings()}")
        print(f"CoTracker loop: {coTracker.loop_rate_timer}")

        [vis.write_result(output_dir) for vis in visualizers]
    finally:
        coTracker.shutdown()


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

        vot_pipeline(
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
