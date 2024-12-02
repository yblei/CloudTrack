"""
semantic_bird_detector base module.
"""

import textwrap
from abc import abstractmethod

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class WrapperBase:
    @abstractmethod
    def run_inference(
        self,
        image_pil: Image,
        prompt: str,
        print_results=True,
        mark_results=True,
    ):
        """Gets an image, returns the results of the inference.

        Args:
            image (Image.PIL): _description_
            print_results (bool, optional): _description_. Defaults to True.
            mark_results (bool, optional): _description_. Defaults to True.

        Returns:
            PIL.Image: The image with the results marked in the form of
                BoundBoxes.
        """
        pass

    def draw_boxes(self, image, boxes, labels):
        max_text_width = 30  # max number of characters in one line

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 10)
        for box, label in zip(boxes, labels):
            label = "\n".join(textwrap.wrap(label, width=max_text_width))
            draw.rectangle(list(box), width=3, outline=(0, 0, 0))
            draw.text(
                (float(box[0]), float(box[1])),
                f"{label}",
                font=font,  # Need to create float here since text modifies the values in place
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )

        return image

    def draw_masks(
        self, image: Image, masks: torch.Tensor, random_color=False
    ):
        image_np = np.array(image)
        # TODO: verwende das hier: https://stackoverflow.com/questions/66095686/apply-a-segmentation-mask-through-opencv

        for mask in masks:
            mask = mask.cpu().numpy()
            if random_color:
                color = np.concatenate([np.random.random(3)], axis=0)
            else:
                # color = np.array([30 / 255, 144 / 255, 255 / 255])
                color = np.array([3, 252, 61])
            h, w = mask.shape[-2:]
            mask_as_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            # overlay mask on image

            # color = np.array([0, 255, 0], dtype="uint8")
            img_with_applied_mask = np.where(
                mask.squeeze(0)[..., None], mask_as_img, image_np
            )
            # img_with_applied_mask = img_with_applied_mask.squeeze(0)
            img_with_applied_mask = img_with_applied_mask.astype(np.uint8)
            image_np = cv2.addWeighted(
                image_np, 0.6, img_with_applied_mask, 0.4, 0
            )

        image = Image.fromarray(image_np.astype(np.uint8))

        return image

    def visualize(
        self, image, boxes=None, labels=None, masks=None, points=None
    ):
        """Visualizes the results of the inference on the image.

        Args:
            image (PIL.Image): The image to visualize the results on.
            boxes (list): A list of boxes in the form of [x1, y1, x2, y2].
            masks (list): A list of masks.
            points (list): A list of points.

        Returns:
            PIL.Image: The image with the results visualized.
        """

        if masks is not None:
            image = self.draw_masks(image, masks)

        if boxes is not None:
            image = self.draw_boxes(image, boxes, labels=labels)

        if points is not None:
            image = self.draw_points(image, points)

        return image

    def get_name(self):
        return self.__class__.__name__

    def get_stats(self) -> str:
        return ""
