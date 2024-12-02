from abc import abstractmethod

import torch


class ResamplingBase:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, points):
        raise NotImplementedError


class Sor(ResamplingBase):
    """Defines statistical outlier removal filter."""

    def __init__(self):
        # everything above 2 std deviations from the mean is considered an outlier
        self.std_dev_mul = 1.7

    def forward(self, points: torch.Tensor):
        """Runs statistical outlier remval filter.

        Args:
            points (torch.tensor): the points to be filtered.
        """

        pts, idx = torch.median(points, dim=1)

        # calculate the distance from the median
        dist = torch.sqrt(torch.sum((points - pts) ** 2, dim=2))

        # calculate the mean of distances
        mean_dist = torch.mean(dist)

        # calculate the standard deviation of distances
        std_dist = torch.std(dist)

        # calculate the threshold
        threshold = mean_dist + self.std_dev_mul * std_dist

        # filter the points
        points = points[dist < threshold].unsqueeze(0)

        return points
