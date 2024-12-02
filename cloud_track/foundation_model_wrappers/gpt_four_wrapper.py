import io
import json
import os
import time
from base64 import b64encode
from hashlib import md5
from pathlib import Path

import requests
from loguru import logger
from PIL import Image

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase


class GPTFourWrapper(WrapperBase):
    def __init__(
        self,
        model="gpt-4o",
        enable_caching=True,
        simulate_time_delay=False,
        system_prompt=None,
        cache_file_name="cache_db.json",
    ):
        """_summary_

        Args:
            model (str, optional): The gpt model to use. Defaults to "gpt-4o".
            enable_caching (bool, optional): Caculate MD5 hash. Don't relaunch query, if result is cached. Defaults to True.
            simulate_time_delay (bool, optional): Simulate the api delay when running on cache mode. Defaults to False.

        Raises:
            ValueError: When the api key is not set.
        """
        # read api from GPT_API_KEY env variable
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise RuntimeError(
                'Could not find API Key in environment variables. Please set the OPENAI_API_KEY environment variable in your .bashrc or .profile like: export OPENAI_API_KEY="XYZXZY...".'
            )

        self.system_prompt = system_prompt
        logger.info(f"Using system prompt: {self.system_prompt}")

        self.model = model
        self.enable_caching = enable_caching
        self.simulate_time_delay = simulate_time_delay

        # check, if api key is none
        if self.api_key is None:
            raise ValueError("Please set the GPT_API_KEY environment variable")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.cache_db_file_path = Path(__file__).parent / cache_file_name

        # if file exists, load the cache
        if self.enable_caching:
            if self.cache_db_file_path.exists():
                with open(self.cache_db_file_path, "r") as f:
                    self.cache_db = json.load(f)
            else:
                self.cache_db = {}

    def encode_image(self, image: Image):
        # convert image to base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        img_str = b64encode(img_byte_arr).decode("utf-8")
        return img_str

    def run_inference(self, prompt: str, image: Image):

        base64_image = self.encode_image(image)

        if self.system_prompt is not None:
            payload = {
                "model": f"{self.model}",
                "messages": [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 300,
            }
        else:
            payload = {
                "model": f"{self.model}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            }

        # caching is enabled: check, if we can return a cached result
        if self.enable_caching:
            # calculate md5 hash of the payload
            payload_hash = md5(json.dumps(payload).encode("utf-8")).hexdigest()

            # check if the payload is in the cache
            lookup_start_time = time.time()
            if payload_hash in self.cache_db.keys():
                response = self.cache_db[payload_hash]["response"]
                api_delay = self.cache_db[payload_hash]["api_delay"]
                lookup_end_time = time.time()
                if self.simulate_time_delay:
                    lookup_time = lookup_end_time - lookup_start_time
                    delta_t = api_delay - lookup_time
                    delta_t = max(0, delta_t)
                    time.sleep(delta_t)
                try:
                    response = response["choices"][0]["message"]["content"]
                    return response
                except KeyError:
                    logger.warning(
                        f"Error in cached response: {response}. Re-running the query."
                    )

        # in any case: send the request to the api
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )

        if self.enable_caching:  # cache the result, if enabled
            api_delay = response.elapsed.total_seconds()
            try:
                response = response.json()
            except requests.exceptions.JSONDecodeError:
                logger.warning(f"Could not parse to json: {response}")
                response = self.get_failed_message(response)

            self.cache_db[payload_hash] = {
                "response": response,
                "api_delay": api_delay,
            }
            self.save_cache()

        try:
            response = response.json()
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Could not parse to json: {response}")
            print("Error in response.")
            response = self.get_failed_message(response)
        try:
            response = response["choices"][0]["message"]["content"]
        except KeyError:
            logger.warning(f"Error in response: {response}")
            response = self.get_failed_message(response)

        return response

    def get_failed_message(self, response):
        return f"""
        Answer: no
        Justification: FAILED - {response}
        """

    def save_cache(self):
        # save the cache to the file
        logger.info(f"Saving cache to {self.cache_db_file_path}")
        with open(self.cache_db_file_path, "w") as f:
            json.dump(self.cache_db, f, indent=4)


if __name__ == "__main__":
    prompt = "We are on a search and rescue mission. Is there an injured person with a gray shirt in the image? If so, please answer yes or no if you would recommend sending a rescue team."

    # load s.png image as PIL from the current directory
    image_folder = Path(__file__).parent / "images"
    model = GPTFourWrapper()

    # for each image in image folder
    answers = []
    for image_path in image_folder.glob("*.png"):
        image = Image.open(image_path)
        res = model.run_inference(prompt, image)
        answers.append(res)

    for ans in answers:
        print(ans["choices"][0]["message"]["content"])
