import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class OpencvModel:
    def __init__(
        self, asset_path: str, url: str, manual_download_required: bool = False
    ):
        """
        Creates a buffered opencv weight. The file is downloaded from the
        given url, if not present in the asset directory.

        Args:
            asset_path (str): The path to the weight.
            url (str): The list of url to download the weight from.
        """
        cloudtrack_folder = Path(__file__).parent.parent.parent.resolve()
        self.model_base_path = cloudtrack_folder / "models" / "opencv"

        self.model_path = self.model_base_path / asset_path
        self.url = url

        # download the file if not present
        if not self.model_path.exists():
            if manual_download_required:
                raise FileNotFoundError(
                    f"Could not find {self.model_path}. Please download it manually as described at the end of this file."
                )
            self.download()

    def download(self):
        # if folders do not exist, create them
        logger.info(f"Downloading {self.url} to {self.model_path.parent}")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(self.url, self.model_path)
        # check, if the file was downloaded
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Could not download {self.url} to {self.model_path}"
            )

    def get_path(self):
        return str(self.model_path)


def set_cuda_params(params):
    """
    Set the tracker to use cuda backend.
    """
    params.backend = cv2.dnn.DNN_BACKEND_CUDA
    params.target = cv2.dnn.DNN_TARGET_CUDA
    return params


def prime_tracker(tracker):
    """
    Run the tracker on a random with a random bounding box to prime the tracker.
    We need to do this since it is pushed to GPU an the first init.
    This is slow in the main loop.
    """
    logger.debug("Initializing Tracker.")
    image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    bbox = (100, 100, 100, 100)
    tracker.init(image, bbox)
    tracker.update(image)

    return tracker


def nano_tracker_factory():
    """
    Creates the Tracker Nano and takes care of the weight downloading.
    """
    backbone_weight_path = "nano/nanotrack_backbone_sim.onnx"
    neckhead_weight_path = "nano/nanotrack_head_sim.onnx"
    backbone_weight_url = "https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx"
    neckhead_weight_url = "https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx"

    backbone_weight = OpencvModel(backbone_weight_path, backbone_weight_url)
    neckhead_weight = OpencvModel(neckhead_weight_path, neckhead_weight_url)

    params = cv2.TrackerNano_Params()
    params.backbone = backbone_weight.get_path()
    params.neckhead = neckhead_weight.get_path()
    # params = set_cuda_params(params)
    return prime_tracker(cv2.TrackerNano_create(params))


def dasiamrpn_tracker_factory():
    """
    Creates the DaSiamRPN Tracker and takes care of weight downloading.
    """
    net_weight_path = "dasiamrpn/dasiamrpn_model.onnx"
    kernel_r1_weight_path = "dasiamrpn/dasiamrpn_kernel_r1.onnx"
    kernel_cls1_weight_path = "dasiamrpn/dasiamrpn_kernel_cls1.onnx"
    net_weight_url = (
        "https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0"
    )
    kernel_r1_weight_url = "https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0"
    kernel_cls1_weight_url = "https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0"

    net_weight = OpencvModel(
        net_weight_path, net_weight_url, manual_download_required=True
    )
    kernel_r1_weight = OpencvModel(
        kernel_r1_weight_path,
        kernel_r1_weight_url,
        manual_download_required=True,
    )
    kernel_cls1_weight = OpencvModel(
        kernel_cls1_weight_path,
        kernel_cls1_weight_url,
        manual_download_required=True,
    )

    params = cv2.TrackerDaSiamRPN_Params()
    params.model = net_weight.get_path()
    params.kernel_r1 = kernel_r1_weight.get_path()
    params.kernel_cls1 = kernel_cls1_weight.get_path()
    params = set_cuda_params(params)
    return prime_tracker(cv2.TrackerDaSiamRPN_create(params))


def vit_tracker_factory():
    """
    Creates the Vit Tracker and takes care of weight downloading.
    """
    model_path = "vit/vitTracker.onnx"
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"

    model = OpencvModel(model_path, model_url)
    params = cv2.TrackerVit_Params()
    params.net = model.get_path()
    params = set_cuda_params(params)
    return prime_tracker(cv2.TrackerVit_create(params))


def goturn_tracker_factory():
    """
    Creates the Goturn Tracker and takes care of weight downloading.
    """

    prototxt_path = "goturn/goturn.prototxt"
    caffemodel_part_paths = [
        f"goturn/goturn.caffemodel.zip.00{idx}" for idx in range(1, 5)
    ]
    caffemodel_path = "goturn/goturn.caffemodel"
    prototxt_url = "https://github.com/opencv/opencv_extra/raw/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking/goturn.prototxt"

    # model consists of 4 parts which must be concatinated.
    caffemodel_urls = [
        f"https://github.com/opencv/opencv_extra/raw/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking/goturn.caffemodel.zip.00{idx}"
        for idx in range(1, 5)
    ]

    prototext = OpencvModel(prototxt_path, prototxt_url)

    # make caffemodel abs path
    caffemodel_path = prototext.model_path.parent / os.path.basename(
        caffemodel_path
    )

    if not os.path.exists(caffemodel_path):
        caffemodels = []
        for path, url in zip(caffemodel_part_paths, caffemodel_urls):
            caffemodels.append(OpencvModel(path, url))

        # run cat goturn.caffemodel.zip* > goturn.caffemodel.zip in goturn folder
        # unzip goturn.caffemodel.zip
        goturn_path = prototext.model_path.parent
        os.system(
            f"cat {goturn_path}/goturn.caffemodel.zip* > {goturn_path}/goturn.caffemodel.zip"
        )
        os.system(f"unzip {goturn_path}/goturn.caffemodel.zip")

        # check, if the file was created successfully
        if not (goturn_path / "goturn.caffemodel").exists():
            # remove all files from folder
            for file in goturn_path.iterdir():
                os.remove(file)
            raise FileNotFoundError(
                f"Could not download and extract {goturn_path}/goturn.caffemodel"
            )

    caffemodel = OpencvModel(caffemodel_path, "Placeholder")

    params = cv2.TrackerGOTURN_Params()
    params.modelTxt = prototext.get_path()
    params.modelBin = caffemodel.get_path()

    params = set_cuda_params(params)

    return prime_tracker(cv2.TrackerGOTURN_create(params))


if __name__ == "__main__":
    tracker = vit_tracker_factory()
    print("vit tracker initialized.")


"""
From https://github.com/opencv/opencv/blob/4.x/samples/python/tracker.py

Tracker demo

For usage download models by following links
For GOTURN:
    goturn.prototxt and goturn.caffemodel: https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking
For DaSiamRPN:
    network:     https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
    kernel_r1:   https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
    kernel_cls1: https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0
For NanoTrack:
    nanotrack_backbone: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx
    nanotrack_headneck: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx

USAGE:
    tracker.py [-h] [--input INPUT] [--tracker_algo TRACKER_ALGO]
                    [--goturn GOTURN] [--goturn_model GOTURN_MODEL]
                    [--dasiamrpn_net DASIAMRPN_NET]
                    [--dasiamrpn_kernel_r1 DASIAMRPN_KERNEL_R1]
                    [--dasiamrpn_kernel_cls1 DASIAMRPN_KERNEL_CLS1]
                    [--dasiamrpn_backend DASIAMRPN_BACKEND]
                    [--dasiamrpn_target DASIAMRPN_TARGET]
                    [--nanotrack_backbone NANOTRACK_BACKEND] [--nanotrack_headneck NANOTRACK_TARGET]
                    [--vittrack_net VITTRACK_MODEL]
"""
