import torch
from PIL import ImageDraw
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from .wrapper_base import WrapperBase


class OwlVitWrapper(WrapperBase):
    def __init__(self, search_query=["bird"], threshold=0.001):
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model = self.model.to("cuda:0")
        self.threshold = threshold
        self.texts = [search_query]
        self.i = 0

    def predict(self, image):
        inputs = self.processor(
            text=self.texts, images=image, return_tensors="pt"
        )
        inputs = inputs.to("cuda:0")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device="cuda:0")
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
        )

        i = self.i
        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )
        return boxes, scores, labels

    def print_results(self, boxes, scores, labels):
        i = self.i
        text = self.texts[i]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {text[label]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

    def mark_results(self, image, boxes, scores, labels):
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle(list(box), width=2)
        return image

    def run_inference(self, image, print_results=True, mark_results=True):
        boxes, scores, labels = self.predict(image)
        if print_results:
            self.print_results(boxes, scores, labels)
        if mark_results:
            image_out = self.mark_results(image, boxes, scores, labels)
        else:
            image_out = image

        return image_out, boxes, scores, labels
