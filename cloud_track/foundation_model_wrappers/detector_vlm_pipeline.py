from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase

from .gpt_four_wrapper import GPTFourWrapper
from .grounding_dino_huggingface_wrapper import GroundingDinoHuggingfaceWrapper
from .grounding_dino_wrapper import GroundingDinoWrapper
from .llava_wrapper import LlavaWrapper
from .paligemma_wrapper import PaligemmaWrapper


class DetectorVlmPipeline(WrapperBase):
    def __init__(
        self,
        vlm: WrapperBase,
        detector: WrapperBase,
        enable_overscan=True,
        overscan_value=50,
    ):
        """
        Instantiates the VLM pipeline. This pipeline combines the GPTFourWrapper
        and the GroundedSamWrapper to find arbitrary objects in images.

        Args:
            role_desctiption (str): Description of the system. e.g. "You are a
                drone on a search and rescue mission." This becomes a part of
                the chatGPT system prompt.
            enable_caching (bool, optional): _description_. Defaults to True.
            simulate_time_delay (bool, optional): _description_. Defaults to False.
            use_sam_hq (bool, optional): _description_. Defaults to True.
            enable_overscan (bool, optional): _description_. Defaults to True.
            gpt_model (str, optional): _description_. Defaults to "gpt-4o".
        """
        self.vlm = vlm
        self.detector = detector

        self.enable_overscan = enable_overscan
        self.overscan_value = 50  # pixels in each direction

        self.debug_enable_gpt = True

    def parse_vlm_response(self, reply):
        """
        In the prompt, we must directly ask for a Help recommendation (yes/no)
        and a Justification.

        Args:
            reply (_type_): _description_

        Returns:
            bool: is this a match with the prompt or not
            justification (str): the justification for the answer
        """

        # split the prompt into lines
        # try:
        #    reply = reply['choices'][0]['message']['content']
        # except KeyError:
        #    logger.warning("Could not parse GPT response. Assuming no match.")
        #    return False, f"Malformed GPT response: {reply}"

        lines = reply.split("\n")
        # find the line with the help recommendation
        help_line = [line for line in lines if "Answer:" in line]
        if not help_line:
            logger.warning(
                "Could not extract help recommendation from the prompt. Assuming no match."
            )
            match = False
            help_line = ["Justification: No help recommendation provided."]

        # extract the help recommendation
        help_line = help_line[0]
        help_recommendation = help_line.split(":")[1].strip().lower()
        if help_recommendation not in ["yes", "no"]:
            logger.warning(
                "Could not parse help recommendation. Assuming no match."
            )
            match = False
        else:
            match = help_recommendation == "yes"  # parse to bool

        # find the line with the justification
        reply = reply.replace("\n", " ")
        if "Justification:" not in reply:
            logger.warning(
                "Could not extract justification from the prompt. Assuming no match."
            )
            justification = "No justification provided."
        else:
            justification = reply.split("Justification:")[1].strip()

        # TODO:
        # Generell:
        # Maximale anzahl GPT Anfragen pro Bild festlegen!!
        # VOT Konfig:
        # Erst über dino scores sortieren - dann der reihe nach mit gpt abfragen
        # bis einer matcht! - den nehmen!
        #
        # SARD Konfig:
        # Alle dino scores abfragen, bis n matches gefunden wurden

        return match, justification

    def parse_vlm_response_list(self, gpt_responses: list[str]):
        """Parases a list of gpt responses. For each response, it parses the

        Args:
            gpt_responses list[str]: A list of GPT responses as text.

        Returns:
            list[bool]: A list of booleans indicating if the response is a match.
            list[str]: A list of justifications for the responses.
        """

        matches = []
        justifications = []

        for response in gpt_responses:
            match, justification = self.parse_vlm_response(response)
            matches.append(match)
            justifications.append(justification)

        return matches, justifications

    def run_inference_inner(
        self, cathegory: str, verbal_description: str, image: Image
    ):

        if self.vlm is None:
            # we give the better prompt to the detector if no VLM is available
            cathegory = verbal_description

        image_pil, masks, boxes_filt, scores = self.detector.run_inference(
            image, prompt=cathegory, mark_results=False
        )

        prompt = verbal_description

        vlm_responses = []
        for row in boxes_filt:
            # crop the object from the image
            # rount row to int
            row = [int(x) for x in row]
            x1, y1, x2, y2 = row

            # overscan the box
            if self.enable_overscan:
                x1 = max(0, x1 - self.overscan_value)
                y1 = max(0, y1 - self.overscan_value)
                x2 = min(image.width, x2 + self.overscan_value)
                y2 = min(image.height, y2 + self.overscan_value)

            cropped_image = image.crop((x1, y1, x2, y2))
            logger.info(f"Found {cathegory} at {row}. Running VLM.")

            # save debug images
            # offload this to the main function
            # cropped_image.save("cropped_image.png")

            if not self.debug_enable_gpt:
                raise NotImplementedError(
                    "GPTFourWrapper is disabled to save tokens during debug."
                )

            logger.info(f"Prompt: {prompt}")
            if self.vlm is not None:
                vlm_response = self.vlm.run_inference(prompt, cropped_image)
            else:
                vlm_response = """
                Answer: Yes
                Justification: No vlm - detector only.
                """
            try:
                logger.info(f"VLM response: {vlm_response}")
            except KeyError:
                logger.warning("VLM Response has unexpected format.")

            vlm_responses.append(vlm_response)

        return vlm_responses, image_pil, masks, boxes_filt, scores

    def run_inference(
        self,
        image: Image,
        category: str,
        description: str = None,
        mark_results=False,
        filter_results=True,
    ):
        """Runs the inference pipeline. Parses the GPT response. If one of the
        objects in the image is a match, returns the bounding box of this
        object.

        Args:
            category (str): The category of the object to find
                (e.g. person, car, etc.)
            description (str): The instruction for the GPT model. (e.g. "Is this a person?")
            image (Image): The image to analyze

        Returns:
            _type_: _description_
        """
        # If no prompt is provided: construct a default prompt
        if not description:
            if " a " in category:
                description = f"Is this {category}?"
            else:
                description = f"Is this a {category}?"

        # If // in the cathegory: split here and use second half for description
        if "//" in category:
            l = category.split("//")
            category = l[0]
            description = l[1]

        # Run the actual inference:
        gpt_responses, image_pil, masks, boxes_filt, scores = (
            self.run_inference_inner(category, description, image)
        )

        # Postprocess the results:
        matches, justifications = self.parse_vlm_response_list(gpt_responses)

        # assemble the labels for the boxes
        labels = []
        for i, match in enumerate(matches):
            if match:
                labels.append(f"{category} (match) | {justifications[i]}")
            else:
                labels.append(f"{category} | {justifications[i]}")

        if mark_results:
            # visualize results
            image_pil = self.visualize(image_pil, boxes_filt, labels=labels)

        if filter_results:
            # filter the boxes, scores, masks, justification based on the matches
            to_filter = [masks, boxes_filt, scores, justifications]
            for idx, list_ in enumerate(to_filter):
                list_ = [list_[i] for i, match in enumerate(matches) if match]
                to_filter[idx] = list_

            masks, boxes_filt, scores, justifications = to_filter

            if len(masks) == 0:
                logger.info("No matches found.")
                return image_pil, None, None, None, None

            boxes_filt = torch.stack(boxes_filt)
            masks = torch.stack(masks)

            masks = None  # not a good solution but fixing the error would take too long
            return image_pil, masks, boxes_filt, scores, justifications

        else:
            return (
                image_pil,
                masks,
                boxes_filt,
                scores,
                matches,
                labels,
                justifications,
            )


def system_prompt_from_description(description: str):
    """
    Helper function to generate a system prompt from a description. Here, we
    conduct some prompt engineering to make the GPT model more effective.
    """

    prompt = f"""{description} Return your answer in the following format:
    Answer: Yes/No
    Justification: [Your justification here]
    """

    return prompt


if __name__ == "__main__":

    output_folder = Path("~/Downloads/flextrack_gpt_output").expanduser()
    output_folder.mkdir(exist_ok=True)

    pipeline = DetectorVlmPipeline()
    opject_cathegory = "person"
    description = "You are looking for a person with a gray shirt, who is missing after beeing injured."
    search_and_rescue_desctiption = "You are on a search and rescue mission. {description} Is this person in the image?"

    image_folder = Path(__file__).parent / "images"

    answers = []
    for image_path in image_folder.glob("*.jpg"):
        logger.info(f"Processing image {image_path.name}")
        image = Image.open(image_path).convert("RGB")  # droop alpha channel

        res = pipeline.run_inference(
            category=opject_cathegory,
            description=search_and_rescue_desctiption,
            image=image,
        )

        image_annotated, *_ = res

        # save the images to the output folder
        image_annotated.save(output_folder / image_path.name)

        # show image with cv2
        cv_image = np.array(image_annotated)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", cv_image)
        cv2.waitKey(10)

        # input("Press Enter to continue...")


def get_detector(detector_model):
    box_threshold = 0.5
    detector_model = detector_model.lower()
    if "sam" in detector_model:
        if "hq" in detector_model:
            use_sam_hq = True
            detector = GroundingDinoWrapper(
                box_threshold=box_threshold, use_sam_hq=use_sam_hq
            )
        elif "lq" in detector_model:
            box_threshold = 0.1
            text_threshold = 0.05
            detector = GroundingDinoHuggingfaceWrapper(
                box_threshold=box_threshold, text_threshold=text_threshold
            )
        else:
            raise ValueError(f"Unknown SAM model {detector_model}.")

    elif "glee" in detector_model:
        split = detector_model.split("_")
        if len(split) == 1:
            raise ValueError(
                f"Unknown GLEE model {detector_model}: Please specify model name like GLEE_[lite/plus/pro]."
            )
        else:
            from .glee_wrapper import GLEEWrapper

            detector = GLEEWrapper(
                model_name=split[1], box_threshold=box_threshold
            )
    else:
        return None
    return detector


def get_vlm(vl_model, system_description, simulate_time_delay):
    if "gpt" in vl_model:
        system_prompt = system_prompt_from_description(system_description)
        vlm = GPTFourWrapper(
            enable_caching=False,
            simulate_time_delay=simulate_time_delay,
            model=vl_model,
            system_prompt=system_prompt,
            cache_file_name="sard_single_shot_cache.json",
        )
    elif "paligemma" in vl_model:
        vlm = PaligemmaWrapper(system_prompt=system_description)
    elif "llava" in vl_model:
        system_prompt = system_description
        vlm = LlavaWrapper(system_prompt=system_prompt, model_name=vl_model)
    else:
        vlm = None
    return vlm


def get_vlm_pipeline(
    vl_model_name: str,
    system_description: str,
    simulate_time_delay: bool,
    detector_name: str,
    openai_api_key: str = None,
):
    """Creates a VLM pipeline with a detector and a VLM model.

    Args:
        vl_model_name (str): Name of the VLM Model. We support:
            - gpt-4o-mini, gpt-4o, gpt-4-turbo
            - llava-hf/llava-1.5-7b-hf, llava-hf/llava-1.5-13b-hf
            - paligemma
        system_description(str)): The system prompt for the VLM model.
        simulate_time_delay (bool): GPT answers can be cached. Retrieving from
            the cache is faster then the API. When simulate_time_delay is True,
            a time.sleep() is applied to simulate the API delay.
        detector_name (str): Name of the detector model. We support:
            - sam_hq, sam_lq
        openai_api_key (str, optional): The openai api key. Defaults to None.

    Raises:
        ValueError: When an unknown model is requested.

    Returns:
        DetectorVLMPipeline: A configured VLM pipeline.
    """
    # models
    vlm = get_vlm(vl_model_name, system_description, simulate_time_delay)
    detector = get_detector(detector_name)

    if detector is None:
        raise ValueError(
            f"Unknown model {detector_name} - cannot run without detector."
        )

    if vlm is None:
        logger.info(
            f"No VLM or unknwon name {vl_model_name} specified. Running without VLM."
        )

    model = DetectorVlmPipeline(vlm, detector, overscan_value=20)

    return model
