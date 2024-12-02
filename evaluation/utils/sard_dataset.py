# taken from here https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard


"""
This file contains functions to add refferals to the VOT 2022 dataset.
"""

import torch
from torch.utils.data import Dataset
import os
from referring_vot import ReferringVot
import torchvision
import cv2

class RefSardDataset:
    def __init__(self, sard_dataset_root):
        # resolve ~
        sard_dataset_root = os.path.expanduser(sard_dataset_root)
        self.vot_dataset_root = sard_dataset_root

        self.referring_vot = ReferringVot(sard_dataset_root)

        # get all the sequences -> get only the non-ambiguous sequences
        self.sequences = self.referring_vot.get_sequences()

        self.sequence_index = 0

    def __iter__(self):
        self.sequence_index = 0
        return self

    def __len__(self):
        return len(self.sequences)

    def __next__(self):
        if self.sequence_index < len(self.sequences):
            sequence = self.sequences[self.sequence_index]
            self.sequence_index += 1

            sequence_root_dir = os.path.join(
                self.vot_dataset_root, "sequences", sequence
            )
            sequence_query = self.referring_vot[sequence]

            return sequence_query, SardSequenceDataset(
                sequence_root_dir, transform=None
            )
        else:
            raise StopIteration


class SardSequenceDataset(Dataset):
    def __init__(self, sequence_root_dir, transform):
        self.sequence_root_dir = sequence_root_dir
        self.transform = transform
        self.gt_file = os.path.join(sequence_root_dir, "groundtruth.txt")
        self.image_dir = os.path.join(sequence_root_dir, "color")
        self.sequence_name = os.path.basename(sequence_root_dir)

        self._data = self.setup_dataset()
        self.idx = 0

    def setup_dataset(self):
        """
        Read all the images and the ground truth file. Create a list of
        dicts: {"image": image, "gt_bbox": gt_bbox}

        image is a string of the path to the image
        gt_bbox is a tensor of 4 elements: x, y, width, height
        """

        with open(self.gt_file, "r") as f:
            lines = f.readlines()
            gt_bboxes = [list(map(float, line.split(","))) for line in lines]

        gt_bboxes = [ltwh_to_xyxy(gt_bbox) for gt_bbox in gt_bboxes]
        gt_bboxes = [torch.tensor(gt_bbox) for gt_bbox in gt_bboxes]

        file_list = os.listdir(self.image_dir)
        file_list.sort()

        images = [
            os.path.join(self.image_dir, image_name)
            for image_name in file_list
        ]
        data = [
            {"image": image, "gt_bbox": gt_bbox}
            for image, gt_bbox in zip(images, gt_bboxes)
        ]

        return data

    def __len__(self):
        return len(os.listdir(self.image_dir))
    
    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self._data):
            raise StopIteration
        sample = self._data[self.idx]
        print(f"Loading image {self.idx} of {len(self._data)}")
        self.idx += 1

        image_path = sample["image"]
        gt_bbox = sample["gt_bbox"]
        # image = self.transform(image)

        # load image with torch
        success = True
        try:
            image = cv2.imread(image_path)
        except Exception as e:
            success = False

        return image, success, gt_bbox

    def get_sequence_name(self):
        return self.sequence_name
    
    def get_resolution(self):
        image = self._data[0]["image"]
        image = cv2.imread(image)
        return image.shape[0], image.shape[1]
    
    def get_sequence_path(self):
        return self.image_dir


def ltwh_to_xyxy(ltwh: list) -> list:
    """
    Convert from (left, top, width, height) to (x_min, y_min, x_max, y_max)
    """
    x, y, w, h = ltwh
    return [x, y, x + w, y + h]


def main():
    r = RefVotDataset("/home/blei/cloud_track/datasets/vot22")

    for sequence_query, sequence_dataset in r:
        print(
            f"{sequence_dataset.get_sequence_name()}" " - " f"{sequence_query}"
        )


if __name__ == "__main__":
    main()
