import io
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from groundingdino.datasets import transforms as T
from PIL import Image

# segment anything
from segment_anything import (
    SamPredictor,
    sam_hq_model_registry,
    sam_model_registry,
)

from .grounding_dino_utils import get_grounding_output, load_model
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
        self.cfg = cfg_in
        self.args = args

        # Grounded_Segment_Anything base director
        gsa_base_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            ),
            "Grounded_Segment_Anything",
        )

        grounding_dino_base_dir = os.path.join(
            gsa_base_dir, "GroundingDINO", "groundingdino"
        )

        # configuration
        self.config_file = os.path.join(
            grounding_dino_base_dir, "config/GroundingDINO_SwinT_OGC.py"
        )
        self.grounded_checkpoint = os.path.join(
            gsa_base_dir,
            "groundingdino_swint_ogc.pth",
        )

        self.sam_tiny_checkpoint = os.path.join(
            gsa_base_dir, "sam_vit_4b8939.pth"
        )

        self.use_sam_hq = use_sam_hq
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = "cuda"

        # setup grounding dino model
        self.model = load_model(
            self.config_file, self.grounded_checkpoint, device=self.device
        )

        # initialize SAM
        if self.use_sam_hq:
            self.sam_version = "vit_b"
            # self.sam_hq_checkpoint = os.path.join(
            #    gsa_base_dir, "sam_vit_h_4b8939.pth"
            # )
            self.sam_hq_checkpoint = os.path.join(
                gsa_base_dir, "sam_vit_b_01ec64.pth"
            )
            self.predictor = SamPredictor(
                sam_hq_model_registry[self.sam_version](
                    checkpoint=self.sam_hq_checkpoint
                ).to(self.device)
            )
        else:
            self.sam_version = "vit_tiny"
            self.predictor = SamPredictor(
                sam_model_registry[self.sam_version](
                    checkpoint=self.sam_tiny_checkpoint
                ).to(self.device)
            )

    def run_inference(
        self,
        image_pil: Image,
        prompt: str,
        print_results=True,
        mark_results=True,
    ):
        """Runs Grounded Sam on the image and returns the results.

        Args:
            image (PIL.Image): The Image to run inference on.
            print_results (bool, optional): Does nothing. Defaults to True.
            mark_results (bool, optional): Does nothing. Defaults to True.
        """
        # run inference on image
        image_pil, boxes_filt, pred_phrases, masks, scores = self.infer(
            image_pil, prompt
        )

        # visualize results
        if mark_results:
            image_pil = self.visualize(
                image_pil, boxes_filt, labels=pred_phrases, masks=masks
            )

        return image_pil, masks, boxes_filt, scores

    def infer(self, image: Image, prompt: str):
        """Runs Grounded Sam on the image and returns the results.

        Args:
            image (PIL.Image): The Image to run inference on.

        Returns:
            image_pil (PIL.Image): The image as a Pillow Image
                (same, that went in).
            boxes_filt (torch.Tensor): The bounding boxes of the objects
                in the image.
            pred_phrases (list): The predicted phrases.
            masks (torch.Tensor): The masks of the objects in the image.

        """
        device = self.device

        warnings.filterwarnings("ignore")
        output_dir = "/tmp/grounded_sam_output"

        # make dir
        os.makedirs(output_dir, exist_ok=True)

        # convert image to pillow and cv2 image
        image_pil, image, image_np = image_to_pil_and_image(image)

        # load model
        model = self.model

        # visualize raw image
        image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

        # run grounding dino model
        boxes_filt, pred_phrases, scores = get_grounding_output(
            model,
            image,
            prompt,
            self.box_threshold,
            self.text_threshold,
            device=self.device,
        )

        image = image_np
        predictor = self.predictor

        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]
        ).to(device)

        masks = None
        if len(boxes_filt) > 0:
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

        return image_pil, boxes_filt, pred_phrases, masks, scores


def main():
    """
    Test configuration to run this wrapper on one image.
    """
    # load image
    image_path = "/home/blei/sam_pt_grounding_dino/Grounded-Segment-Anything/"
    "assets/bussard/0035.jpg"
    image_pil = Image.open(image_path).convert("RGB")  # load image

    # initialize wrapper
    wrapper = GroundedSamWrapper()

    # run inference
    loaded_image, image_pil = wrapper.run_inference(image_pil)

    # show results
    plt.imshow(loaded_image)
    plt.show()

    # save results
    loaded_image.save("/tmp/output.png")


if __name__ == "__main__":
    main()
