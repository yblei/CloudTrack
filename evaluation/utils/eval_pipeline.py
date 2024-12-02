from .aggregation_tools import (
    load_multirun,
    apply_to_level_multirun,
    apply_to_sequence_in_multirun,
    average_key,
    apply_to_level_inner,
    apply_to_level_run,
)
from .metrics import (
    frame_iou,
    mean_fps,
    count_frames_in_sequence,
    sum_frames,
    add_frame_to_motmetrics,
)

from pathlib import Path
import motmetrics as mm
import numpy as np
import matplotlib.pyplot as plt


def average_iou(sequence_data):
    return average_key(sequence_data, "iou")


def average_fps(run_data):
    return average_key(run_data, "fps")

def average_free_fps(run_data):
    return average_key(run_data, "free_fps", ignore_zeros=True)

def mean_fps_no_fm_frames(run_data):
    return mean_fps(run_data,  only_non_fm_frames=True)


def multirun_pipeline(path: Path):
    """
    Loads the runs from the multirun directory and applies all aggregation
    functions.

    Args:
        path (Path): Path to the lop level directory of the runs.

    Returns:
        dict: Dictionary contianing the aggregated data.
    """
    # Load the multirun data
    multirun_data = load_multirun(path)

    # Apply the aggregation functions
    # IoU
    multirun_data = apply_to_level_multirun(
        multirun_data, func=frame_iou, level="frame", key_name="iou"
    )  # calculate iou for every frame

    multirun_data = apply_to_level_multirun(
        multirun_data, func=average_iou, level="sequence", key_name="iou"
    )  # calculate mean iou for every sequence
    multirun_data = apply_to_level_multirun(
        multirun_data, func=average_iou, level="run", key_name="iou"
    )  # calculate mean iou for every run

    # FPS
    multirun_data = apply_to_sequence_in_multirun(
        multirun_data, mean_fps, "fps"
    )  # calculate fps for every sequence
    
    multirun_data = apply_to_sequence_in_multirun(
        multirun_data, mean_fps_no_fm_frames, "free_fps"
    )  # calculate mean fps for every run
    
    print("make free fps")

    multirun_data = apply_to_level_multirun(
        multirun_data, func=average_fps, level="run", key_name="fps"
    )  # calculate mean fps for every run
    
    multirun_data = apply_to_level_multirun(
        multirun_data, func=average_free_fps, level="run", key_name="free_fps"
    )  # calculate mean fps for every sequence

    # frames
    multirun_data = apply_to_level_multirun(
        multirun_data,
        func=count_frames_in_sequence,
        level="sequence",
        key_name="frames",
    )  # calculate number of frames for every sequence
    multirun_data = apply_to_level_multirun(
        multirun_data, func=sum_frames, level="run", key_name="frames"
    )  # calculate number of frames for every run

    # MOT metrics
    # Create an accumulator that will be used to calculate the metrics

    print("WARNING: TAKE CARE OF NOT DETECTED OBJECTS")
    results = {}

    for run_name, run_data in multirun_data.items():
        flextrack_acc = mm.MOTAccumulator(auto_id=True)
        results[run_name] = flextrack_acc

        def add_frame_to_motmetrics_local(frame_data):
            return add_frame_to_motmetrics(frame_data, flextrack_acc)

        apply_to_level_run(
            run_data,
            func=add_frame_to_motmetrics_local,
            level="frame",
            key_name="mot_acc",
        )  # calculate mot metrics for every frame

    mh = mm.metrics.create()
    summary = mh.compute(
        flextrack_acc, metrics=["motp", "num_frames"], name="Flextrack"
    )

    names = []
    accumulators = []
    for run_name, acc in results.items():
        names.append(run_name)
        accumulators.append(acc)

    summary = mh.compute_many(
        accumulators, metrics=["motp", "num_frames", "mota"], names=names
    )

    # add to the multirun_dict
    for run_name, acc in results.items():
        multirun_data[run_name]["motp"] = summary["motp"][str(run_name)]

    mot_metrics_summary = summary

    return multirun_data, mot_metrics_summary


def gather_data_from_runs(multirun_data, relevant_keys):
    """Get the relevant keys from the runs and returns them in a dictionary.

    Args:
        multirun_data (_type_): _description_
        relevant_keys (_type_): _description_

    Returns:
        _type_: _description_
    """
    dict_out = {}
    #number_of_runs = len(multirun_data.keys())
    number_of_runs = max([int(run) for run in multirun_data.keys()]) + 1

    # make empty dict with the keys
    for run in range(number_of_runs):
        for relevant_key in relevant_keys:
            if str(run) not in dict_out.keys():
                dict_out[str(run)] = {}
            dict_out[str(run)][relevant_key] = None

    # Print the results
    for run in range(number_of_runs):
        # print(f"Run {run}:")
        try:
            for relevant_key in relevant_keys:
                #print(f"    {relevant_key}: {multirun_data[str(run)][relevant_key]}")
                dict_out[str(run)][relevant_key] = multirun_data[str(run)][relevant_key]
                dict_out[str(run)]["hydra_config"] = multirun_data[str(run)]["hydra_config"]
        except KeyError:
            # print warning about missing run
            print(f"Run {run} not found.")
        #print("\n")

    return dict_out


def make_plot(sensitivity_study_performance, key_x, key_y, ax_glob, add_to_global: bool = False):
    # scatter plot 
    def dict_to_np_array(dict_in, key):
        """
        Converts a dictionary with arrays to a numpy array.
        """
        x = []
        for run in sorted(dict_in.keys()):
            x.append(dict_in[run][key])
        return np.array(x)

    # generate legend
    legend = []
    for run in sorted(sensitivity_study_performance.keys()):
        legend.append(sensitivity_study_performance[run]["hydra_config"]["cv_tracker"]["name"])

    x = dict_to_np_array(sensitivity_study_performance, key_x)
    y = dict_to_np_array(sensitivity_study_performance, key_y)

    # Local Diagram
    plt.xlabel(key_x)
    plt.ylabel(key_y)
    for i, txt in enumerate(legend):
        #if not i%1 == 0:
        #    continue
        print(f"Adding {txt} to the plot.")
        plt.scatter(x[i], y[i], label=txt)
    plt.title("Re-Initialize depending on threshold: Sensitivity study")
    plt.legend()
    plt.show()

    if add_to_global:
        # Global Diagram
        ax_glob.scatter(x, y)
        for i, txt in enumerate(legend):
            ax_glob.annotate(txt, (x[i], y[i]), textcoords='offset pixels',
            ha='left',va='bottom')