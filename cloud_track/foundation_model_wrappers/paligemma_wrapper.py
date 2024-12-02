from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase


class PaligemmaWrapper(WrapperBase):
    def __init__(
        self,
        model="google/paligemma-3b-mix-224",
        enable_caching=True,
        simulate_time_delay=False,
        system_prompt=None,
    ):
        # model_id = "google/paligemma-3b-mix-448-keras"
        self.max_new_tokens = 20
        self.system_prompt = system_prompt
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model)

    def run_inference(self, prompt: str, image: Image):
        prompt = self.system_prompt + prompt
        logger.info(f"Prompt: {prompt}")
        inputs = self.processor(prompt, image, return_tensors="pt")
        inputs.to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
            decoded_output = self.processor.decode(
                outputs[0], skip_special_tokens=True
            )[len(prompt) :]

        decoded_output.replace("\n", "")

        yes_no = "undefined"
        if "yes" in decoded_output.lower():
            yes_no = "Yes"
        elif "no" in decoded_output.lower():
            yes_no = "No"

        answer_formatted = (
            f"Answer: {yes_no} \n Justification: {decoded_output}"
        )

        return answer_formatted


if __name__ == "__main__":
    system_prompt = "You are on a search and rescue mission. Answer in the following format: Answer: Yes/No"
    prompt = "Would you recommend sending a rescue team?"

    # load s.png image as PIL from the current directory
    image_path = (
        "/home/blei/cloud_track/outputs/2024-07-30/15-46-57/cropped_image.png"
    )
    model = PaligemmaWrapper(system_prompt=system_prompt)

    # for each image in image folder
    image = Image.open(image_path).convert("RGB")
    print(model.run_inference(prompt, image))
