import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    pipeline,
)

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase


class LlavaWrapper(WrapperBase):
    def __init__(
        self,
        model_name="llava-hf/llava-1.5-7b-hf",
        enable_caching=True,
        simulate_time_delay=False,
        system_prompt=None,
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #############
        # self.model = LlavaForConditionalGeneration.from_pretrained(
        #    model_name).to(self.device)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.pipe = pipeline(
            "image-to-text",
            model=model_name,
            model_kwargs={"quantization_config": quantization_config},
        )

        ###############
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.system_prompt = system_prompt

    def run_inference(self, prompt: str, image: Image):
        if not self.system_prompt:
            raise ValueError("System prompt is not set.")

        prompt_1_final = f"USER: <image>\n {self.system_prompt} ASSISTANT:"

        ans = self.run_inference_inner(prompt_1_final, image)
        answer_1 = ans.split("ASSISTANT:")[-1].strip()

        prompt_2 = f"USER: <image>\n{self.system_prompt} ASSISTANT: {answer_1}</s>USER: {prompt} ASSISTANT:"
        ans = self.run_inference_inner(prompt_2, image)
        answer_2 = ans.split("ASSISTANT:")[-1].strip()

        answer_2_yes_no = answer_2.lower().strip()
        if "yes" in answer_2_yes_no:
            answer_2_yes_no = "yes"
        elif "no" in answer_2_yes_no:
            answer_2_yes_no = "no"

        ans_formatted = (
            f"Answer: {answer_2_yes_no} \nJustification: {answer_1}"
        )

        return ans_formatted

    def run_inference_inner(self, prompt, image):
        # inputs = self.processor(text=prompt, images=image,
        #                        return_tensors = "pt")
        # inputs.to(self.device)

        # Generate
        # with torch.inference_mode():
        # generate_ids = self.model.generate(**inputs, max_new_tokens=30)

        #   ans = self.processor.batch_decode(
        #      generate_ids, skip_special_tokens=True,
        #     clean_up_tokenization_spaces=False)

        ans = self.pipe(
            image, prompt=prompt, generate_kwargs={"max_new_tokens": 30}
        )

        return ans[0]["generated_text"]


if __name__ == "__main__":
    system_prompt = "You are an intelligent assistant, helping a drone on a search and rescue mission. Describe the image."
    llava = LlavaWrapper(system_prompt=system_prompt)

    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    # url = "https://www.scienceabc.com/wp-content/uploads/2018/09/injured-man.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    image = Image.open(
        "/home/blei/cloud_track/cloud_track/foundation_model_wrappers/images/cropped_image.png"
    )
    prompt = "Based on this information, would you recommend sending help? Answer with yes or no."

    ans = llava.run_inference(prompt, image)
    print(ans)
