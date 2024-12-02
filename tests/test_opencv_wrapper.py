import unittest
from cloud_track.tracker_wrapper.opencv_wrapper import OpenCVWrapper
import torch
import numpy as np


class TestOpenCVWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # create a random image
        images = []
        for i in range(10):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            images.append(image)

        self.images = images
        self.bbox = torch.Tensor([0, 0, 100, 100])

    def test_InitNano(self):
        """
        Tests, if nanotrack works out of the box. It should download the
        model by itself and return a model object.
        """
        model = OpenCVWrapper("nano")
        self.assertIsNotNone(model)

    def test_TrackNano(self):
        """
        Tests, if nanotrack works out of the box. It should download the
        model by itself and return a model object.
        """
        model = OpenCVWrapper("nano")
        model.init(self.images[0], self.bbox)
        for image in self.images[1:]:
            tmp_box = model.update(image)
        self.assertIsNotNone(tmp_box)
