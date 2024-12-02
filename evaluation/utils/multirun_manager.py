from .single_shot_pipeline import ExperimentPipeline
from dataclasses import dataclass
from pathlib import Path
import tabulate
import numpy as np
# import pythonTexTools as ptt


@dataclass
class EvalResult:
    # pipeline
    pipeline: ExperimentPipeline

    # detection performance
    mAP: float
    pr_curve_precision: list[float]
    pr_curve_recall: list[float]
    overall_precision: float
    overall_recall: float
    fps: float

    # size dependent performance
    sizes: list[str]  # the names of the sizes
    recall_sizes: list[float]


class MultirunManager:
    def __init__(self, base_path: Path, injury_annotation_file: Path):
        self.experiment_pipelines = []

        self.__load_experiments(base_path, injury_annotation_file)
        self.experiment_results = []

    def __load_experiments(self, base_path: Path, injury_annotation_file: Path):
        for path in base_path.iterdir():
            if not path.is_dir():
                continue
            # logger.info(f"Loading pipeline from {path}")
            self.experiment_pipelines.append(
                ExperimentPipeline(path, injury_annotation_file))

    def get_model_headers_exps(self, mode):
        if mode == "ours":
            models = ["gpt", "13b", "7b", "paligemma"]
            headers = ["Exp", "GPT-4-mini P", "GPT-4-mini R", "GPT-4-mini [AP]", "llava 13b P", "llava 13b R", "llava 13b [AP]", "llava 7b P", "llava 7b R", "llava 7b [AP]",
                       "Paligemma P", "Paligemma R", "Paligemma [AP]"]

        elif mode == "baseline":
            models = ["sam", "lite", "plus", "pro"]
            headers = ["Exp", "SAM P", "SAM R", "SAM [AP]", "glee lite P", "glee lite R", "glee lite [AP]",
                       "glee plus P", "glee plus R", "glee plus [AP]", "glee pro P", "glee pro R", "glee pro [AP]"]
        else:
            raise ValueError("Please specify a mode. baseline or ours")

        exps = ["person", "shirt_gray", "shirt_green", "shirt_blue",
                "pose_laying", "pose_standing", "pose_sitting", "injury"]

        return models, headers, exps

    def get_and_print_results(self, iou_threshold: float = 0.5, conf_threshold: float = -1, mode=None):
        models, headers, exps = self.get_model_headers_exps(mode)

        res = {}
        for exp in exps:
            res[exp] = {}
            for model in models:
                for pipeline in self.experiment_pipelines:
                    if exp not in pipeline.name:
                        continue
                    if model not in pipeline.name:
                        continue
                    
                    print("#----------------------------------------#")
                    print("Pipeline: ", pipeline.name)

                    p, r = pipeline.get_precision_recall(
                        iou_threshold=iou_threshold, conf_threshold=conf_threshold)
                    ap = pipeline.get_average_precision(
                        iou_threshold=iou_threshold)
                    t_obj = pipeline.get_time_per_object()
                    t_f = pipeline.get_time_per_frame()
                    # delete the last line

                    # store results in percent
                    res[exp][model] = {
                        "name": pipeline.name,
                        "experiment": exp,
                        "model": model,
                        "p": p,
                        "r": r,
                        "ap": ap,
                        "t_obj": t_obj,
                        "t_f": t_f
                    }

                    print("\r", end="")
                    print(
                        f"Precision: {res[exp][model]['p']}, Recall: {res[exp][model]['r']}, AP: {res[exp][model]['ap']}")

        return res

    def print_result_table(self, res, mode=None):

        models, headers, exps = self.get_model_headers_exps(mode)

        # print as a table - values are in percent
        print("#----------------------------------------#")
        table = []
        map_dict = {}
        tf_dict = {}
        t_obj_dict = {}
        for exp in exps:
            row = [exp]
            for model in models:
                if model not in map_dict:
                    map_dict[model] = []
                    tf_dict[model] = []
                    t_obj_dict[model] = []
                p = res[exp][model]["p"]
                row.append(str(round(p*100, 2)) + "%")
                r = res[exp][model]["r"]
                row.append(str(round(r*100, 2)) + "%")
                ap = res[exp][model]["ap"]
                row.append(str(round(ap*100, 2)) + "%")
                map_dict[model].append(ap)
                tf_dict[model].append(res[exp][model]["t_f"])
                t_obj_dict[model].append(res[exp][model]["t_obj"])
            table.append(row)
        

        # add the mAP
        row = ["mAP"]
        for model in models:
            for i in range(2):
                row.append(" --- ")
            row.append(str(round(np.average(map_dict[model])*100,2)) + "%")
        table.append(row)
        
        for d, name in zip([tf_dict, t_obj_dict], ["Time per frame [s]", "Time per object [s]"]):
            row = [name]
            for model in models:
                for i in range(2):
                    row.append(" --- ")
                row.append(str(round(np.average(d[model]), 3)) + "s")
            
            table.append(row)

        print(tabulate.tabulate(table, headers,
              tablefmt="grid", numalign="center"))

        return table, headers

    def analysis(self):
        for pipeline in self.experiment_pipelines:
            self.experiment_results.append(self.analysis_inner(pipeline))

    def analysis_inner(self, pipeline: ExperimentPipeline) -> EvalResult:
        # we need the following information:
        # - precision
        # - recall

        precisions, recalls = pipeline.get_precision_recall_curve()

        overall_precision = pipeline.get_precision()
        overall_recall = pipeline.get_recall()

        bins, size_recalls = pipeline.get_size_recalls()

        result = EvalResult()

        result.pipeline = pipeline
        result.pr_curve_precision = precisions
        result.pr_curve_recall = recalls
        result.overall_precision = overall_precision
        result.overall_recall = overall_recall

        result.sizes = bins
        result.recall_sizes = size_recalls

        return result
