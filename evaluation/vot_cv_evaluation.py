#from __future__ import annotations
from utils.vot_dataset import RefVotDataset
from utils import fix_logging
import hydra
from omegaconf import DictConfig
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cloud_track.utils.flow_control import LoopRateTimer, PerformanceTimer
from cloud_track.pipeline.single_shot_detection import (
    single_shot_detection_inner_cv,
)
#from cloud_track.foundation_model_wrappers import GroundedSamWrapper
from cloud_track.tracker_wrapper.co_tracker_wrapper import (
#    CoTrackerWrapper,
    visualizer,
)
from cloud_track.tracker_wrapper.opencv_wrapper import OpenCVWrapper
from cloud_track.rpc_communication import RpcWrapper
from loguru import logger


def vot_pipeline(
    dataset,
    prompt: str,
    tracker_cfg,
    vLModel,
    resolution: tuple = None,
    vis_cfg: bool = False,
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

    tracker = OpenCVWrapper(tracker_cfg.name, reinit_threshold=tracker_cfg.threshold)

    try:
        visualizers = []
        if vis_cfg.write_video:
            visualizers.append(
                visualizer.Visualizer(no_window=vis_cfg.no_windows, fps=visualizer_fps)
                )
        visualizers.append(
            visualizer.gt_writer(no_window=vis_cfg.no_windows, fps=visualizer_fps)
        )

        loop_rate_timer = PerformanceTimer()
        fm_timer = PerformanceTimer()

        single_shot_detection_inner_cv(
            prompt,
            vLModel,
            dataset,
            tracker,
            visualizers,
            loop_rate_timer,
            fm_timer,
            f_max=f_max,
            adaptive_fps=True,
            reinit_each=reinit_each,
            output_dir=output_dir,
        )

        print(f"FM: {fm_timer}")
        print(f"Main Loop: {loop_rate_timer}")
        print(f"Average FPS: {1/loop_rate_timer.get_average()}")
        print(vLModel.get_stats())


        [vis.write_result(output_dir) for vis in visualizers]
    finally:
        tracker.shutdown()

@hydra.main(config_path="../conf", config_name="ref_vot_cv_evaluation")
@logger.catch # this must be second!!!
def main(cfg: DictConfig):
    """Run evaluation on VOT22 dataset.

    Args:
        cfg (DictConfig): The hydra config.
    """
    fix_logging()
    vot_root = cfg.dataset.vot_root
    platform = cfg.platform
    backend = cfg.backend
    cv_tracker = cfg.cv_tracker

    vis_cfg = cfg.visualizer
    f_max = cfg.f_max
    reinit_each = cfg.reinit_each
    #gpt_model = cfg.gpt_model

    ref_vot_dataset = RefVotDataset(vot_root)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for sequence_query, sequence_dataset in ref_vot_dataset:
        print(sequence_query)
                # make directory
        sequence_dir = os.path.join(hydra_dir, sequence_dataset.sequence_name)
        os.makedirs(sequence_dir, exist_ok=True)
        logger.info(f"Creating output directory: {sequence_dir}")
        if platform.use_rpc:
            model = RpcWrapper(backend.rpc_ip, backend.rpc_port)
        else:
            #from cloud_track.foundation_model_wrappers import GroundedSamWrapper
            #model = GroundedSamWrapper(use_sam_hq=True)
            from cloud_track.foundation_model_wrappers import DetectorVlmPipeline
            model = DetectorVlmPipeline(use_sam_hq=True, model_id=gpt_model)


        # resolution = cfg.source.resolution

        # print sequence name
        print(f"Processing: {sequence_dataset.sequence_name}")

        vot_pipeline(
            dataset=sequence_dataset,
            prompt=sequence_query,
            vLModel=model,
            vis_cfg=vis_cfg,
            fps=4,
            interactive=False,
            f_max=f_max,
            output_dir=sequence_dir,
            reinit_each=reinit_each,
            tracker_cfg=cv_tracker
        )


if __name__ == "__main__":
    main()

