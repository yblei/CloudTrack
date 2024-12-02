from pathlib import Path
import json
import yaml
import numpy as np


def load_sequence(sequence_path: Path) -> dict:
    """
    Load the sequence data from the sequence_path folder.
    Args:
        sequence_path: Path to the sequence folder which contains the track.json file.

    Returns a dictionary with frame_id, box, gt and timestamp.
    """

    with open(sequence_path / "track.json") as f:
        data = json.load(f)
    return data


def load_run(run_path: Path):
    """
    Walk over all sequences in run_path folder.
    Args:
        run_path: Path to the run folder which contains the sequence folders.

    Returns a dictionary with sequence names as keys and sequence data as values.
    """
    out = {}
    for sequence_path in run_path.iterdir():
        if sequence_path.is_dir() and not "." in sequence_path.name:
            sequence_name = sequence_path.name
            # load the sequence config from the .hydra folder

            sequence_data = {
                "raw_data": load_sequence(sequence_path),
            }
            out[sequence_name] = sequence_data

        with open(run_path / ".hydra" / "config.yaml") as f:
            sequence_config = yaml.safe_load(f)

        run_data = {"raw_data": out, "hydra_config": sequence_config}

    return run_data


def load_multirun(multirun_path: Path):
    """
    Walk over all directoried in multirun_path folder with contains the
    folders with the individual runs.

    Args:
        multirun_path: Path to the multirun folder which contains the run folders.

    Returns a dictionary with run names as keys and run data as values.
    """
    out = {}
    for run_path in multirun_path.iterdir():
        if run_path.is_dir():
            if not "." in run_path.name:
                run_name = run_path.name

                run_data = load_run(run_path)
                out[run_name] = run_data
    return out


## Apply function to every sequence in a run
def apply_to_run(run_data, func, key_name):
    for sequence_name, sequence_data in run_data["raw_data"].items():
        run_data["raw_data"][sequence_name][key_name] = func(sequence_data)
    return run_data


def apply_to_sequence_in_multirun(multirun_data, func, key_name):
    for run_name, run_data in multirun_data.items():
        multirun_data[run_name] = apply_to_run(run_data, func, key_name)
    return multirun_data


def apply_to_level_multirun(multirun_data, func, level: str, key_name: str):
    """
    Applies func to every entry in level and adds the result as a new key
    to this entry.

    Args:
        multirun_data (_type_): _description_
        func (_type_): _description_
        level (str): _description_
        key_name (str): _description_

    Returns:
        _type_: _description_
    """
    name_to_level = {  # raw_data subdir is expanded automatically
        "multirun": 0,
        "run": 1,
        "sequence": 2,
        "frame": 3,
    }
    level = name_to_level[level]

    return apply_to_level_inner(multirun_data, func, level, key_name)


def apply_to_level_run(run_data, func, level, key_name):
    name_to_level = {
        "run": 0,
        "sequence": 1,
        "frame": 2,
    }

    return apply_to_level_inner(run_data, func, name_to_level[level], key_name)


def apply_to_level_inner(multirun_data, func, level, key_name):
    if type(multirun_data) == dict:
        if "raw_data" in multirun_data.keys():
            multirun_data_content = multirun_data["raw_data"]
        else:
            multirun_data_content = multirun_data

    if level == 0:
        multirun_data[key_name] = func(multirun_data_content)
    else:
        level = level - 1
        if type(multirun_data_content) == dict:
            for run_name, run_data in multirun_data_content.items():
                apply_to_level_inner(run_data, func, level, key_name)
        if type(multirun_data_content) == list:
            for run_data in multirun_data_content:
                apply_to_level_inner(run_data, func, level, key_name)

    return multirun_data


def average_key(run_data, key_name, ignore_zeros=False):
    """
    Averages the key_name over all elements in run_data.
    """
    if type(run_data) == dict:
        values = [
            sequence_data[key_name] for sequence_data in run_data.values()
        ]
    elif type(run_data) == list:
        values = [sequence_data[key_name] for sequence_data in run_data]
    else:
        raise ValueError("run_data must be a dict or a list.")
    
    if ignore_zeros:
        values = [value for value in values if value != 0]
        
    # remove all nan
    values = [value for value in values if not np.isnan(value)]

    return np.mean(values)


def sum_key(run_data, key_name):
    """
    Sums the key_name over all elements in run_data.
    """
    if type(run_data) == dict:
        values = [
            sequence_data[key_name] for sequence_data in run_data.values()
        ]
    elif type(run_data) == list:
        values = [sequence_data[key_name] for sequence_data in run_data]
    else:
        raise ValueError("run_data must be a dict or a list.")

    return np.sum(values)
