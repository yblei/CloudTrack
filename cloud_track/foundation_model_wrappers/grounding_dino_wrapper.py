from pathlib import Path

import groundingdino
import numpy as np
import torch
from groundingdino.datasets import transforms as T
from groundingdino.util.inference import load_model, predict
from PIL import Image

from .wrapper_base import WrapperBase


def image_to_pil_and_image(image_pil: Image):
    """Combatibility to old grounded sam demo.

    Args:
        image_path (PIL.Image): Pillow Image.add()

    Returns:
        PIL.Image: The image as a Pillow Image (same, that went in).add()

    """

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w

    image_np = np.array(image_pil)

    return image_pil, image, image_np


def pil_to_dino(image_pil: Image):
    """Convert a PIL Image for usage with dino Image and return it"""

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_transformed, _ = transform(image_pil, None)
    return image_transformed


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class GroundingDinoWrapper(WrapperBase):
    def __init__(
        self,
        cfg_in=None,
        use_sam_hq=True,
        box_threshold=0.3,
        text_threshold=0.25,
        args="",
    ) -> None:

        cloudtrack_folder = Path(__file__).parent.parent.parent.resolve()
        dino_model_folder = cloudtrack_folder / "models/groundingdino"

        dino_pth_path = dino_model_folder / "groundingdino_swint_ogc.pth"

        dino_module_folder = Path(groundingdino.__file__).parent.resolve()
        dino_ogc_path = (
            dino_module_folder / "config/GroundingDINO_SwinT_OGC.py"
        )

        self.model = load_model(
            model_config_path=str(dino_ogc_path),
            model_checkpoint_path=str(dino_pth_path),
        )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def run_inference(
        self,
        image_pil: Image,
        prompt: str,
        print_results=True,
        mark_results=False,
    ):
        """Runs Grounded Sam on the image and returns the results.

        Args:
            image (PIL.Image): The Image to run inference on in RGB.
            print_results (bool, optional): Does nothing. Defaults to True.
            mark_results (bool, optional): Does nothing. Defaults to True.
        """
        image_transformed = pil_to_dino(image_pil)

        boxes, scores, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # masks = [torch.Tensor([False]) for _ in range(len(boxes))]
        # masks = torch.zeros((len(boxes), 1, 1))

        # make a bool tensor of shape (len(boxes), 1, image_h, image_w)
        masks = torch.zeros(
            (len(boxes), 1, image_pil.size[1], image_pil.size[0])
        )

        # make this tensor all false
        masks = masks.bool()

        # visualize results
        if mark_results:
            image_pil = self.visualize(
                image_pil, boxes, labels=phrases, masks=masks
            )

        if boxes is not None:
            boxes = boxes.cpu().numpy()

            w, h = image_pil.size
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h

            # convert cxcywh to x1y1x2y2
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            print(boxes, scores, phrases)

            boxes = torch.tensor(boxes)

            scores = scores.cpu().numpy()
            # convert to list of floats
            scores = [float(score) for score in scores]

        return image_pil, masks, boxes, scores
