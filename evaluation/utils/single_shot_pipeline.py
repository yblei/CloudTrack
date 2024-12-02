from pathlib import Path
import json
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from .metrics import get_tp_fp_tn_fn
from tabulate import tabulate
from tqdm import tqdm
import yaml
from loguru import logger
from hashlib import md5


class InjuryEvaluator:
    def __init__(self, annotation_file: Path):
        self.injury_annotations = json.load(open(annotation_file))

        # set target mapping -> MUST BE THE SAME AS THE ONE USED IN THE LABLING TOOL!!!
        self.target_mapping = {
            "laying_down": True,
            "not_defined": True,
            "null": True,
            "Running": False,
            "seated": True,
            "stands": False,
            "Walking": False,
        }

    def is_injured(self, frame, object_idx, class_name):
        if not self.target_mapping[class_name]:
            return False

        frame = f"{frame}.jpg"
        is_injured = self.injury_annotations[frame][str(object_idx)]["injured"]
        return is_injured

    def __call__(self, frame, object_idx, class_name):
        return self.is_injured(frame, object_idx, class_name)


class ShirtColorEvaluator:
    def __init__(self, annotation_file: Path):
        self.cloth_annotations = json.load(open(annotation_file))

    def get_shirt_color(self, frame, object_idx):
        frame = f"{frame}.jpg"
        try:
            color = self.cloth_annotations[frame][str(
                object_idx)]["shirt_color"]
        except KeyError:
            logger.warning(
                f"No shirt color found for {frame} {object_idx} - this should not happen!")
            return "UNKNOWN"
        return color


class PoseEvaluator:
    def __init__(self, annotation_file: Path):
        self.pose_annotations = json.load(open(annotation_file))

    def get_pose(self, frame, object_idx):
        frame = f"{frame}.jpg"
        try:
            pose = self.pose_annotations[frame][str(
                object_idx)]["class_name"]
        except KeyError:
            logger.warning(
                f"No pose found for {frame} {object_idx} - this should not happen!")
            return "UNKNOWN"
        return pose


class ExperimentPipeline:
    def __init__(self, result_path: Path, injury_annotation_file: Path):
        self.frames, self.res = self.__load_results(result_path)
        self.injury_evaluator = InjuryEvaluator(injury_annotation_file)
        self.shirt_color_evaluator = ShirtColorEvaluator(
            injury_annotation_file)
        self.pose_evaluator = PoseEvaluator(injury_annotation_file)

        vl_model = self.res["metadata"]["vl_model"]
        detector_model = self.res["metadata"]["detector"]

        self.__parse_hydra_dir(result_path)

        self.name = f"{vl_model}_{detector_model}/{self.exp_name}"
        
        # create cache file in cwd
        self.cache_db_file_path = Path.cwd() / "cache_db.json"
        
        # load cache if it exists or create a new one
        if self.cache_db_file_path.exists():
            self.cache_db = json.load(open(self.cache_db_file_path))
        else:
            self.cache_db = {}
            # write empty cache to disk
            json.dump(self.cache_db, open(self.cache_db_file_path, "w"))

        self.__set_bin_sizes()
        self.__set_sizes()
        self.__add_gt_info()
        self.__fix_types()
        
    def check_gt_is_positive(self, gt_entry: dict):
        if self.exp_name == "sar_injury":
            return gt_entry["injured"]
        elif self.exp_name == "sar_shirt":
            res = "gray" in gt_entry["shirt_color"].lower()
            if not res:
                # logger.info(f"shirt color {gt_entry['shirt_color'].lower()} is not gray.")
                pass
            return res
        elif self.exp_name == "sar_person":
            return True
        elif self.exp_name == "sar_shirt_green":
            return "green" in gt_entry["shirt_color"].lower()
        elif self.exp_name == "sar_shirt_blue":
            return "blue" in gt_entry["shirt_color"].lower()
        elif self.exp_name == "sar_shirt_gray":
            return "gray" in gt_entry["shirt_color"].lower()
        elif self.exp_name == "sar_pose_laying":
            return "laying_down" in gt_entry["pose"].lower()
        elif self.exp_name == "sar_pose_sitting":
            return "seated" in gt_entry["pose"].lower()
        elif self.exp_name == "sar_pose_standing":
            return "stands" in gt_entry["pose"].lower()
        else:
            raise ValueError(f"Unknown experiment name {self.exp_name}")

    def check_detection_is_positive(self, detection_entry: dict):
        if "injured" in detection_entry:
            return detection_entry["injured"]
        elif "match" in detection_entry:
            return detection_entry["match"]

    def get_confusion_matrix(self, iou_threshold=0.5, size_select="default", skip_vlm_check=False, conf_threshold=-1):
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for frame in self.frames:
            gt_boxes = []
            pred_boxes = []

            # retrieve all injured annotations
            for idx, gt in enumerate(self.res[frame]["annotation"]):
                if size_select != "default" and "size" in gt:
                    if size_select != gt["size"]:
                        continue
                if self.check_gt_is_positive(gt) or skip_vlm_check:
                    box = gt["bndbox"]
                    box = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
                    gt_boxes.append(box)

            # retrieve all injured predictions
            for result in self.res[frame]["result"]:
                if self.check_detection_is_positive(result) or skip_vlm_check:
                    if result["score"] > conf_threshold:
                        pred_boxes.append(result["box"])

            if False:
                # remove duplicates with iou > 0.5
                for idx, box in enumerate(pred_boxes):
                    for idx2, box2 in enumerate(pred_boxes):
                        if idx == idx2:
                            continue
                        if box is None or box2 is None:
                            continue
                        if get_tp_fp_tn_fn([box], [box2], iou_threshold=iou_threshold)[0] > 0:
                            pred_boxes[idx] = None

                pred_boxes = [box for box in pred_boxes if box is not None]

            f_tp, f_fp, f_tn, f_fn = get_tp_fp_tn_fn(
                gt_boxes, pred_boxes, iou_threshold=iou_threshold)

            tp += f_tp
            fp += f_fp
            tn += f_tn
            fn += f_fn

        return tp, fp, tn, fn

    def print_confusion_matrix(self, iou_threshold=0.5):
        tp, fp, tn, fn = self.get_confusion_matrix(iou_threshold=iou_threshold)

        tn = "N/A"

        # table
        table = [[f"TP {tp}", f"FP {fp}"], [f"FN {fn}", f"TN {tn}"]]

        # print confusion matrix
        print(tabulate(table))
        
    def get_md5_state(self, additional_params: dict= None):
        """
        Returns the md5 hash of the state and the additional params.
        """
        time = 0
        for frame in self.frames:
            time += abs(self.res[frame]["time_s"])
        
        to_hash = {
            "additional_params": additional_params,
            "frames": len(self.frames),
            "pipeline_name": self.name,
            "total_time": time
            }
        return md5(json.dumps(to_hash).encode("utf-8")).hexdigest()

    def get_precision_recall_curve(self, iou_threshold=0.5, skip_vlm_check=False):
        function_args = {
            "iou_threshold": iou_threshold,
            "skip_vlm_check": skip_vlm_check,
        }
        hash = self.get_md5_state(function_args)

        if hash in self.cache_db:
            print(f"Using cached version for {self.name}.")
            return self.cache_db[hash]["precisions"], self.cache_db[hash]["recalls"]
        else:
            precisions = []
            recalls = []
            for conf_threshold in tqdm(np.linspace(0, 1, 100)): # chenge last to 100!!
                p, r = self.get_precision_recall(
                    iou_threshold=iou_threshold, skip_vlm_check=skip_vlm_check, conf_threshold=conf_threshold)
                precisions.append(p)
                recalls.append(r)
                
            self.cache_db[hash] = {
                "precisions": precisions,
                "recalls": recalls
            }
            self.write_cache()

            return precisions, recalls

    def write_cache(self):
        # load the cach from file then merge it with the current cache
        file_cache = json.load(open(self.cache_db_file_path))
        self.cache_db.update(file_cache)
        
        
        print("Writing cache to disk.")
        json.dump(self.cache_db, open(self.cache_db_file_path, "w"))

    def get_precision_recall(self, iou_threshold=0.5, skip_vlm_check=False, conf_threshold=-1):
        delta = 0.00000000000000001
        tp, fp, tn, fn = self.get_confusion_matrix(
            iou_threshold=iou_threshold, skip_vlm_check=skip_vlm_check, conf_threshold=conf_threshold)
        p = tp / (tp + fp + delta)
        r = tp / (tp + fn + delta)
        return p, r

    def get_precision(self, iou_threshold=0.5, skip_vlm_check=False):
        # precision mit size select macht keinen Sinn, da wir eine große
        # Anzahl an falsch positiven haben, die nicht in die Größe fallen.

        tp, fp, _, _ = self.get_confusion_matrix(
            iou_threshold=iou_threshold, skip_vlm_check=skip_vlm_check)
        return tp / (tp + fp)

    def get_recall(self, iou_threshold=0.5, size_select="default", skip_vlm_check=False):
        tp, _, _, fn = self.get_confusion_matrix(
            iou_threshold=iou_threshold, size_select=size_select, skip_vlm_check=skip_vlm_check)
        return tp / (tp + fn)

    def get_time_per_frame(self):
        time = 0
        for frame in self.frames:
            time += abs(self.res[frame]["time_s"])
        return time / len(self.frames)

    def get_time_per_object(self):
        time = 0
        num_detections = 0
        for frame in self.frames:
            num_detections += len(self.res[frame]["result"])
            time += abs(self.res[frame]["time_s"])

        return time / num_detections

    def get_dataset_statistics(self):
        num_injured = 0
        num_not_injured = 0

        for frame in self.frames:
            for idx, obj in enumerate(self.res[frame]["annotation"]):
                if obj["injured"]:
                    num_injured += 1
                else:
                    num_not_injured += 1

        total = num_injured + num_not_injured

        return total, num_injured, num_not_injured

    def print_dataset_statistics(self):
        total, num_injured, num_not_injured = self.get_dataset_statistics()
        print(f"Total number of people: {total}")
        print(
            f"Number of injured: {num_injured} ({num_injured/total*100:.2f}%)")
        print(
            f"Number of not injured: {num_not_injured} ({num_not_injured/total*100:.2f}%)")

    def print_precision_recall(self, iou_threshold=0.5, skip_vlm_check=False, conf_threshold=-1):
        # precision mit size select macht keinen Sinn, da wir eine große
        # Anzahl an falsch positiven haben, die nicht in die Größe fallen.
        # precision_val = self.get_precision(
        #    iou_threshold=iou_threshold, skip_injury_check=skip_injury_check)
        # recall_val = self.get_recall(
        #    iou_threshold=iou_threshold, skip_injury_check=skip_injury_check)

        precision_val, recall_val = self.get_precision_recall(
            iou_threshold=iou_threshold, skip_vlm_check=skip_vlm_check, conf_threshold=conf_threshold)
        average_precision = self.get_average_precision(
            iou_threshold=iou_threshold, skip_vlm_check=skip_vlm_check)

        print(f"Precision: {precision_val*100:.2f}%")
        print(f"Recall: {recall_val*100:.2f}%")
        print(f"Average Precision: {average_precision*100:.2f}%")

    def smooth_pr_curve(self, precisions, recalls):
        # create smooted 11 point curve
        n = np.linspace(0, 1, 11)
        smoothed_precisions = []
        smoothed_recalls = []
        for r_threshold in n:
            # find the maximum precision for all recalls >= r_threshold
            smoothed_precisions.append(
                max([p for r, p in zip(recalls, precisions) if r >= r_threshold]+[0]))
            smoothed_recalls.append(r_threshold)

        return smoothed_precisions, smoothed_recalls
    
    def get_average_precision(self, iou_threshold=0.5, skip_vlm_check=False):
        precisions, recalls = self.get_precision_recall_curve(
            iou_threshold=iou_threshold, skip_vlm_check=skip_vlm_check)
        smoothed_precisions, smoothed_recalls = self.smooth_pr_curve(
            precisions, recalls)
        return np.mean(smoothed_precisions)

    def make_precision_recall_plot(self, iou_threshold=0.5, skip_vlm_check=False):
        precisions, recalls = self.get_precision_recall_curve(
            iou_threshold=iou_threshold, skip_vlm_check=skip_vlm_check)

        smoothed_precisions, smoothed_recalls = self.smooth_pr_curve(
            precisions, recalls)

        print("\r")

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        ax.plot(recalls, precisions)
        ax.plot(smoothed_recalls, smoothed_precisions, marker="o")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        # print max precision
        print(f"Max precision: {max(precisions)}")
        print(f"Max recall: {max(recalls)}")
        print("Average precision: ", np.mean(precisions))
        return fig

    def make_size_plot(self):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        ax.hist(self.list_of_diagonals, bins=50)
        # set title
        ax.set_title("Histogram of diagonal lengths")
        ax.set_xlabel("Diagonal length")
        ax.set_ylabel("Count")

        # make vertical lines for the thresholds
        ax.axvline(self.thresholds[0], color="red", linestyle="--")
        ax.axvline(self.thresholds[1], color="red", linestyle="--")

        return fig

    def print_size_statistics(self):
        for bin in self.bins:
            print(
                f"Bin {bin}: {len([d for d in self.list_of_diagonals if self.bins[bin][0] <= d <= self.bins[bin][1]])} elements")
            # print bounds
            print(
                f"Lower bound: {int(self.bins[bin][0])}. Upper bound: {int(self.bins[bin][1])}")

    def print_size_recalls(self, iou_threshold=0.5):
        recalls, sizes = self.get_size_recalls(iou_threshold=iou_threshold)
        for recall, size in zip(recalls, sizes):
            print(f"Recall for size {size}: {recall*100:.2f}%")

    def get_size_recalls(self, iou_threshold=0.5):
        recalls = []
        bins = []
        for size in self.bins:
            recall = self.get_recall(
                size_select=size, iou_threshold=iou_threshold)
            recalls.append(recall)
            bins.append(size)
        return recalls, bins

    def print_full_report(self, iou_threshold=0.5):
        print(f"Experiment: {self.name}")
        print("")
        print("Detector Performance in terms of class (i.e. in terms of 'person'):")
        self.print_precision_recall(
            skip_vlm_check=True, iou_threshold=iou_threshold)
        print("")
        print("Performance after semantic classification (i.e. in terms of 'person' & 'is injured'):")
        self.print_precision_recall(iou_threshold=iou_threshold)
        print("")
        print("Confusion Matrix:")
        self.print_confusion_matrix(iou_threshold=iou_threshold)
        print("--------------------")
        print("Size dependent performance:")
        print("")
        self.print_size_recalls(iou_threshold=iou_threshold)
        print("--------------------")
        print("Time Information:")
        print("")
        print(f"Average time per frame: {self.get_time_per_frame():.2f}s")
        print(f"Average time per object: {self.get_time_per_object():.2f}s")

    def print_dataset_report(self):
        print("Dataset Information:")
        print("")
        self.print_size_statistics()
        print("")
        self.print_dataset_statistics()

    def __parse_hydra_dir(self, result_path: Path):
        hydra_dir = result_path / ".hydra"
        if not hydra_dir.exists():
            logger.error(
                f"Could not find .hydra directory in {result_path}. Using injury experiment name.")
            self.exp_name = "sar_injury"
            return

        # open config.yaml
        config_file = hydra_dir / "config.yaml"
        config = yaml.full_load(config_file.open())

        self.exp_name = config["experiment"]

    def __load_results(self, result_path: Path):
        res_file = result_path / "sard_single_shot.json"
        log_path = result_path / "sard_single_shot.log"

        # the result if valid, if "Done." is in the last line of the log file
        # raise an error if this is not the case

        with log_path.open() as f:
            lines = f.readlines()
            if "Done." not in lines[-1]:
                raise ValueError(
                    f"Experiment {result_path} did not finish successfully. "
                    "Check the log file.")

        res = json.load(res_file.open())

        frames = list(res.keys())
        try:
            frames.remove("metadata")
        except ValueError:
            pass

        try:
            frames.remove("prompt")
        except ValueError:
            pass

        return frames, res

    def __set_bin_sizes(self):
        list_of_diagonals = []

        for frame in self.frames:
            for idx, object in enumerate(self.res[frame]["annotation"]):
                if self.injury_evaluator(frame, idx, object["name"]):
                    bndbox = object["bndbox"]
                    width = int(bndbox["xmax"]) - int(bndbox["xmin"])
                    height = int(bndbox["ymax"]) - int(bndbox["ymin"])
                    diagonal = np.sqrt(width**2 + height**2)
                    list_of_diagonals.append(diagonal)

        # group in 3 bins (small, medium, large). each bin should have an equal
        # number of elements
        list_of_diagonals.sort()
        n = len(list_of_diagonals)
        small = list_of_diagonals[:n//3]
        medium = list_of_diagonals[n//3:2*n//3]
        large = list_of_diagonals[2*n//3:]

        small_upper_bound = max(small)
        small_lower_bound = min(small)
        medium_upper_bound = max(medium)
        medium_lower_bound = min(medium)
        large_upper_bound = max(large)
        large_lower_bound = min(large)

        small_medium_threshold = (small_upper_bound + medium_lower_bound) / 2
        medium_large_threshold = (medium_upper_bound + large_lower_bound) / 2

        bins = {
            "small": [small_lower_bound-2, small_medium_threshold],
            "medium": [small_medium_threshold, medium_large_threshold],
            # add tolerance here for numerical errors
            "large": [medium_large_threshold, large_upper_bound+2],
        }

        # set the bin sizes
        self.bins = bins
        self.list_of_diagonals = list_of_diagonals
        self.thresholds = [small_medium_threshold, medium_large_threshold]

    def __set_sizes(self):
        bins = self.bins
        for frame in self.frames:
            for idx, object in enumerate(self.res[frame]["annotation"]):
                if self.injury_evaluator(frame, idx, object["name"]):
                    bndbox = object["bndbox"]
                    width = int(bndbox["xmax"]) - int(bndbox["xmin"])
                    height = int(bndbox["ymax"]) - int(bndbox["ymin"])
                    diagonal = np.sqrt(width**2 + height**2)
                    size = "undefined"
                    if bins["small"][0] <= diagonal <= bins["small"][1]:
                        size = "small"
                    elif bins["medium"][0] <= diagonal <= bins["medium"][1]:
                        size = "medium"
                    elif bins["large"][0] <= diagonal <= bins["large"][1]:
                        size = "large"
                    else:
                        # das kann passieren, wenn ein nicht verletzter Mensch gruppiert werden soll
                        print(
                            f"WARNING: Diagonal {diagonal} did not fit into any bin")
                    self.res[frame]["annotation"][idx]["size"] = size

    def __add_gt_info(self):
        for frame in self.frames:
            for idx, object in enumerate(self.res[frame]["annotation"]):
                self.res[frame]["annotation"][idx]["injured"] = self.injury_evaluator(
                    frame, idx, object["name"])
                self.res[frame]["annotation"][idx]["shirt_color"] = self.shirt_color_evaluator.get_shirt_color(
                    frame, idx)
                self.res[frame]["annotation"][idx]["pose"] = self.pose_evaluator.get_pose(
                    frame, idx)

    def __fix_types(self):
        """
        Convert every number that is string to float.
        """

        for frame in self.frames:
            for idx, object in enumerate(self.res[frame]["annotation"]):
                for key in object["bndbox"]:
                    object["bndbox"][key] = float(object["bndbox"][key])
