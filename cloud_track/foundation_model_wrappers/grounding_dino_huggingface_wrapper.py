import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .wrapper_base import WrapperBase


class GroundingDinoHuggingfaceWrapper(WrapperBase):
    def __init__(
        self,
        cfg_in=None,
        use_sam_hq=True,
        box_threshold=0.3,
        text_threshold=0.25,
        args="",
    ):
        model_id = "IDEA-Research/grounding-dino-tiny"

        self.device = "cuda"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)

    def run_inference(
        self,
        image_pil: Image,
        prompt: str,
        print_results=True,
        mark_results=True,
    ):
        # convert pil image to tensor
        image = pil_to_tensor(image_pil).unsqueeze(0).to(self.device)
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size()[::-1]],
        )

        results = results[0]
        masks = None
        scores = results["scores"].cpu().numpy()
        boxes_filt = results["boxes"].cpu().numpy()

        return image_pil, masks, boxes_filt, scores
