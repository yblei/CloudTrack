import numpy as np
import motmetrics as mm
from collections import namedtuple
from .aggregation_tools import sum_key

def frame_iou(frame_data):
    """
    Calculates the iou for the frame. Assuming only one ground truth and one detection.
    """
    ground_truth = frame_data["gt"]
    box = frame_data["box"]
    return iou(ground_truth, box)

def mean_frame_iou(frame_data, iou_threshold=0.5):
    """
    Calculate mean IoU for a frame. Multiple objects can be present. 
    The IoU is calculated for each object and the mean is returned.    

    Args:
        frame_data (_type_): _description_
    """
    
    ground_truth = frame_data["gt"]
    boxes = frame_data["boxes"]
    
    matched_boxes, iou_values = match_box(ground_truth, boxes, iou_threshold=iou_threshold) # false negatives count as 0 IoU -> This is already in the iou_values
    
    # false positives count as 0 IoU -> we add those numbers to the iou_values
    num_true_positives = sum([iou > 0 for iou in iou_values])
    num_false_positives = len(boxes) - num_true_positives
    
    iou_values += [0] * num_false_positives
    
    return np.mean(iou_values)
    
def ground_truth_rate(frame_data, iou_threshold=0.5):
    """Ground truth rate is the rate of ground truth objects that are detected.

    Args:
        frame_data (_type_): _description_
    """
    
    ground_truth = frame_data["gt"]
    boxes = frame_data["boxes"]
    
    matched_boxes, iou_values = match_box(ground_truth, boxes, iou_threshold=iou_threshold)
    
    num_true_positives = sum([iou > iou_threshold for iou in iou_values])
    
    
    return num_true_positives / len(ground_truth)


def get_tp_fp_tn_fn(ground_truth, boxes, iou_threshold=0.5):
    """Calcualte the number of true positives, false positives, true negatives and false negatives for a frame.

    Args:
        frame_data (_type_): _description_
        iou_threshold (float, optional): _description_. Defaults to 0.5.
    """

    matched_boxes, iou_values, duplicate_detections = match_box(ground_truth, boxes, iou_threshold=iou_threshold)
    
    num_true_positives = sum([iou > iou_threshold for iou in iou_values])
    num_false_positives = len(boxes) - num_true_positives
    num_false_negatives = len(ground_truth) - num_true_positives
    num_true_negatives = 0
    
    # duplicate detections are no false positives in our analysis.
    num_false_positives -= sum(duplicate_detections)
    
    return num_true_positives, num_false_positives, num_true_negatives, num_false_negatives


def precision(frame_data, iou_threshold=0.5):
    """Calculate the precision for a frame.

    Args:
        frame_data (_type_): _description_
        iou_threshold (float, optional): _description_. Defaults to 0.5.
    """

    num_true_positives, num_false_positives, _, _ = get_tp_fp_tn_fn(frame_data, iou_threshold=iou_threshold)
    
    if num_true_positives + num_false_positives == 0:
        return 0
    
    return num_true_positives / (num_true_positives + num_false_positives)

def recall(frame_data, iou_threshold=0.5):
    """Calculate the recall for a frame.

    Args:
        frame_data (_type_): _description_
        iou_threshold (float, optional): _description_. Defaults to 0.5.
    """

    num_true_positives, _, _, num_false_negatives = get_tp_fp_tn_fn(frame_data, iou_threshold=iou_threshold)
    
    if num_true_positives + num_false_negatives == 0:
        return 0
    
    return num_true_positives / (num_true_positives + num_false_negatives)

def match_box(gt_list, box_list, iou_threshold=0.5):
    """
    Match the boxes to the ground truth.
    """

    rectangle = namedtuple("rectangle", ["x_min", "y_min", "x_max", "y_max"])
    gt_list = [rectangle(*gt) for gt in gt_list]
    box_list = [rectangle(*box) for box in box_list]

    matched_boxes = []
    iou_values = []
    duplicate_detections = []
    for gt in gt_list:
        best_iou = 0
        best_box = None
        num_duplicate_detections = 0
        for box in box_list:
            iou_val = iou(gt, box)
            if iou_val > best_iou:
                best_iou = iou_val
                # especially report duplicate detections on one ground truth
                if best_box is not None and iou_val > iou_threshold:
                    num_duplicate_detections += 1
                best_box = box
        if best_iou > iou_threshold:
            matched_boxes.append(best_box)
            iou_values.append(best_iou)
            duplicate_detections.append(num_duplicate_detections)
        else:
            matched_boxes.append(None)
            iou_values.append(0)
            duplicate_detections.append(0)

    return matched_boxes, iou_values, duplicate_detections


def iou(ground_truth, box):
    """
    Calculate the intersection over union of two boxes.
    """

    rectangle = namedtuple("rectangle", ["x_min", "y_min", "x_max", "y_max"])
    ground_truth = rectangle(
        ground_truth[0], ground_truth[1], ground_truth[2], ground_truth[3]
    )

    if box is None:  # if no box is found, return an IoU of 0
        return 0

    box = rectangle(box[0], box[1], box[2], box[3])

    # calculate the intersection
    xA = max(ground_truth.x_min, box.x_min)  # top left x
    yA = max(ground_truth.y_min, box.y_min)  # top left y
    xB = min(ground_truth.x_max, box.x_max)  # bottom right x
    yB = min(ground_truth.y_max, box.y_max)  # bottom right y

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # calculate the area of the boxes
    boxArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    gtArea = (ground_truth[2] - ground_truth[0] + 1) * (
        ground_truth[3] - ground_truth[1] + 1
    )

    # calculate the union
    iou = interArea / float(boxArea + gtArea - interArea)

    return iou


def mean_fps(sequence_data, only_non_fm_frames=False):
    """
    Adds the fps to every sequence, when parsed to apply_to_sequence_multirun.
    Applies the mean backend timestamp difference to the sequence data.
    only_non_fm_frames: If True, only non fm frames are considered for the fps calculation.
    """

    timestamps = np.array([])
    is_fm_frame = []
    for frame in sequence_data["raw_data"]:
        timestamps = np.append(timestamps, frame["timestamp"])
        is_fm_frame.append(frame["is_fm_frame"])

    # calculate fps as mean of the timestamps differences
    diff = np.diff(timestamps)
    
    # we have no time for the first frame. We thus remove the first is_tm_frame
    is_fm_frame = is_fm_frame[1:]
    
    # whenever we have a frame, that is an fm frame, we need to set the time difference to 1.258104196803438 (average backend response time)
    diff_out = []
    for diff, is_fm in zip(diff, is_fm_frame):
        if only_non_fm_frames and is_fm:
            continue
        if is_fm:
            diff_out.append(1.258104196803438)
        else:
            diff_out.append(diff)
    diff = np.array(diff_out)
    
    fps = 1 / np.mean(diff)

    return fps


def count_frames_in_sequence(sequence_data):
    """
    Counts the number of frames in the sequence.
    """
    return len(sequence_data)


def sum_frames(run_data):
    return sum_key(run_data, "frames")


def add_frame_to_motmetrics(frame_data, mot_acc: mm.MOTAccumulator):
    rectangle = namedtuple("rectangle", ["x_min", "y_min", "x_max", "y_max"])
    ground_truth = frame_data["gt"]
    ground_truth = rectangle(
        ground_truth[0], ground_truth[1], ground_truth[2], ground_truth[3]
    )
    box = frame_data["box"]

    gt_obj_id = 1  # since we only have one gt and one detection

    if box is None:
        mot_acc.update(
            [gt_obj_id],
            [],
            [
                [],
            ],
        )
        return None

    box = rectangle(box[0], box[1], box[2], box[3])

    # eucledian distance between gt center and detection center
    gt_center = np.array(
        [
            ground_truth.x_min + (ground_truth.x_max - ground_truth.x_min) / 2,
            ground_truth.y_min + (ground_truth.y_max - ground_truth.y_min) / 2,
        ]
    )
    detection_center = np.array(
        [
            box.x_min + (box.x_max - box.x_min) / 2,
            box.y_min + (box.y_max - box.y_min) / 2,
        ]
    )
    distance_obj_to_hyp = np.sqrt(np.sum((gt_center - detection_center) ** 2))

    detection_obj_id = 1

    mot_acc.update(
        [gt_obj_id],
        [detection_obj_id],
        [
            [distance_obj_to_hyp],
        ],
    )

    return None
