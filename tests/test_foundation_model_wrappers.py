import unittest
from cloud_track.foundation_model_wrappers.grounding_dino_huggingface_wrapper import GroundingDinoHuggingfaceWrapper
from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import get_vlm_pipeline
from cloud_track.foundation_model_wrappers.gpt_four_wrapper import GPTFourWrapper
from cloud_track.foundation_model_wrappers.llava_wrapper import LlavaWrapper


class TestFoundationModelWrappers(unittest.TestCase):
    def test_grounding_dino_huggingface_wrapper(self):
        model = GroundingDinoHuggingfaceWrapper()
        self.assertIsNotNone(model)

    def test_GPT_four_wrapper(self):
        model = GPTFourWrapper()
        self.assertIsNotNone(model)

    def test_llava_wrapper(self):
        model = LlavaWrapper()
        self.assertIsNotNone(model)

    def test_detector_vlm_pipeline(self):
        # make a gpt four model
        vlm = "llava-hf/llava-1.5-7b-hf"
        # make a grounding dino model
        dino = "sam_lq"

        model = get_vlm_pipeline(
            vlm, "Empty system prompt", False, dino
        )

        self.assertIsNotNone(model)
