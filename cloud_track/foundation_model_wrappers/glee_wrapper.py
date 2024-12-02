# parts of this code are taken from the GLEE demo code

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from loguru import logger
from PIL import Image

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase

try:
    GLEE_DEMO_PATH = Path("~/GLEE_demo").expanduser()
    # check, if folder exists
    if not GLEE_DEMO_PATH.exists():
        raise ImportError(f"GLEE model path does not exist: {GLEE_DEMO_PATH}")

    # add path to python path
    sys.path.append(str(GLEE_DEMO_PATH))
    try:
        from GLEE.glee.models.glee_model import GLEE_Model
    except AttributeError as e:
        logger.info(
            "If you get the missing Linear atribute error -"
            "pip intall Pillow==9.5.0"
        )
        raise e
    from GLEE.glee.config import add_glee_config
    from GLEE.glee.config_deeplab import add_deeplab_config

    GLEE_IMPORTED = True
except ImportError as e:
    logger.info("failed to import GLEE model.")
    GLEE_IMPORTED = False
    # log the message
    logger.info(e.msg)


try:
    from detectron2.config import get_cfg
except ImportError:
    logger.info("failed to import detectron2.")


class GLEEWrapper(WrapperBase):
    def __init__(self, model_name="lite", box_threshold=0.2):
        self.model_name = model_name
        self.box_threshold = box_threshold

        if not GLEE_IMPORTED:
            raise ImportError(
                "Cannot run this part of the piplien with importing GLEE. "
                "Fix the import error first."
            )
        # set device
        if torch.cuda.is_available():
            print("use cuda")
            self.device = "cuda"
        else:
            print("use cpu")
            self.device = "cpu"

        # set topK
        self.topK_instance = 20

        # load model
        self.glee_demo_root = GLEE_DEMO_PATH
        # backup cwd and set to self.glee_demo_root
        cwd = Path.cwd()
        os.chdir(self.glee_demo_root)
        self.model, self.inference_type = self.load_model(model_name)

        # set resizer
        inference_size = 800
        self.size_divisibility = 32
        if self.inference_type != "LSJ":
            self.resizer = torchvision.transforms.Resize(
                inference_size, antialias=True
            )
        else:
            self.resizer = torchvision.transforms.Resize(
                size=1535, max_size=1536, antialias=True
            )

        # set normalizer
        self.set_normalizer()

        # restore cwd
        os.chdir(cwd)

    def set_normalizer(self):
        pixel_mean = (
            torch.Tensor([123.675, 116.28, 103.53])
            .to(self.device)
            .view(3, 1, 1)
        )
        pixel_std = (
            torch.Tensor([58.395, 57.12, 57.375]).to(self.device).view(3, 1, 1)
        )

        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def load_model(self, mode):
        model = None
        if mode.lower() == "lite":
            print("use GLEE-Lite (ResNet50)")
            inference_type = "resize_shot"

            cfg_r50 = get_cfg()
            add_deeplab_config(cfg_r50)
            add_glee_config(cfg_r50)
            conf_files_r50 = "GLEE/configs/R50.yaml"
            checkpoints_r50 = torch.load("GLEE_R50_Scaleup10m.pth")
            cfg_r50.merge_from_file(conf_files_r50)
            GLEEmodel_r50 = GLEE_Model(
                cfg_r50, None, self.device, None, True
            ).to(self.device)
            GLEEmodel_r50.load_state_dict(checkpoints_r50, strict=False)
            GLEEmodel_r50.eval()

            model = GLEEmodel_r50

        elif mode.lower() == "plus":
            inference_type = "resize_shot"
            print("use GLEE-Lite (Swin Transformer)")
            cfg_swin = get_cfg()
            add_deeplab_config(cfg_swin)
            add_glee_config(cfg_swin)
            conf_files_swin = "GLEE/configs/SwinL.yaml"
            checkpoints_swin = torch.load("GLEE_SwinL_Scaleup10m.pth")
            cfg_swin.merge_from_file(conf_files_swin)
            GLEEmodel_swin = GLEE_Model(
                cfg_swin, None, self.device, None, True
            ).to(self.device)
            GLEEmodel_swin.load_state_dict(checkpoints_swin, strict=False)
            GLEEmodel_swin.eval()

            model = GLEEmodel_swin

        elif mode.lower() == "pro":
            print("use GLEE-Pro")
            inference_type = "LSJ"
            cfg_eva02 = get_cfg()
            add_deeplab_config(cfg_eva02)
            add_glee_config(cfg_eva02)
            conf_files_eva02 = "GLEE/configs/EVA02.yaml"
            checkpoints_eva = torch.load("GLEE_EVA02_Scaleup10m.pth")
            cfg_eva02.merge_from_file(conf_files_eva02)
            GLEEmodel_eva02 = GLEE_Model(
                cfg_eva02, None, self.device, None, True
            ).to(self.device)
            GLEEmodel_eva02.load_state_dict(checkpoints_eva, strict=False)
            GLEEmodel_eva02.eval()

            model = GLEEmodel_eva02

        else:
            raise ValueError(f"Mode {mode} not supported.")

        return model, inference_type

    def run_inference(
        self,
        image_pil: Image,
        prompt: str,
        print_results=True,
        mark_results=True,
    ):
        expressiong = prompt
        image_np = np.asarray(image_pil)
        image_torch = torch.as_tensor(
            np.ascontiguousarray(image_np.transpose(2, 0, 1))
        )
        image_torch = self.normalizer(image_torch.to(self.device))[None,]

        _, _, ori_height, ori_width = image_torch.shape

        if self.inference_type == "LSJ":
            resize_image = self.resizer(image_torch)
            image_size = torch.as_tensor(
                (resize_image.shape[-2], resize_image.shape[-1])
            )
            re_size = resize_image.shape[-2:]
            infer_image = torch.zeros(1, 3, 1536, 1536).to(image_torch)
            infer_image[:, :, : image_size[0], : image_size[1]] = resize_image
            padding_size = (1536, 1536)
        else:
            resize_image = self.resizer(image_torch)
            image_size = torch.as_tensor(
                (resize_image.shape[-2], resize_image.shape[-1])
            )
            re_size = resize_image.shape[-2:]
            if self.size_divisibility > 1:
                stride = self.size_divisibility
                # the last two dims are H,W, both subject to divisibility
                # requirement
                padding_size = (
                    (image_size + (stride - 1)).div(
                        stride, rounding_mode="floor"
                    )
                    * stride
                ).tolist()
                infer_image = torch.zeros(
                    1, 3, padding_size[0], padding_size[1]
                ).to(resize_image)
                infer_image[0, :, : image_size[0], : image_size[1]] = (
                    resize_image
                )

        prompt_list = {"grounding": [expressiong]}
        with torch.no_grad():
            (outputs, _) = self.model(
                infer_image,
                prompt_list,
                task="grounding",
                batch_name_list=[],
                is_train=False,
            )

        mask_pred = outputs["pred_masks"][0]
        mask_cls = outputs["pred_logits"][0]
        boxes_pred = outputs["pred_boxes"][0]

        scores = mask_cls.sigmoid().max(-1)[0]
        scores_per_image, topk_indices = scores.topk(
            self.topK_instance, sorted=True
        )

        pred_class = mask_cls[topk_indices].max(-1)[1].tolist()
        pred_boxes = boxes_pred[topk_indices]

        boxes = LSJ_box_postprocess(
            pred_boxes, padding_size, re_size, ori_height, ori_width
        )
        mask_pred = mask_pred[topk_indices]

        if prompt == "person":
            print("ACHTUNG: DER PROMPT IST IMMERNOCH DIE KATHETORIE PERSON")

        # convert to list
        scores_per_image = scores_per_image.tolist()

        boxes = boxes.int()

        boxes_filt = []
        scores_filt = []
        for box, score in zip(boxes, scores_per_image):
            if score > self.box_threshold:
                boxes_filt.append(box)
                scores_filt.append(score)

        return image_pil, None, boxes_filt, scores_filt


def LSJ_box_postprocess(out_bbox, padding_size, crop_size, img_h, img_w):
    # postprocess box height and width
    boxes = box_cxcywh_to_xyxy(out_bbox)
    lsj_sclae = torch.tensor(
        [padding_size[1], padding_size[0], padding_size[1], padding_size[0]]
    ).to(out_bbox)
    crop_scale = torch.tensor(
        [crop_size[1], crop_size[0], crop_size[1], crop_size[0]]
    ).to(out_bbox)
    boxes = boxes * lsj_sclae
    boxes = boxes / crop_scale
    boxes = torch.clamp(boxes, 0, 1)

    scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
    scale_fct = scale_fct.to(out_bbox)
    boxes = boxes * scale_fct
    return boxes


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


if __name__ == "__main__":
    # test
    wrapper = GLEEWrapper(model_name="lite")
    del wrapper
    wrapper = GLEEWrapper(model_name="plus")
    del wrapper
    wrapper = GLEEWrapper(model_name="pro")
    print(f"Using: {wrapper.device}")
