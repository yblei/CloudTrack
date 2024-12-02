import time
from contextlib import ContextDecorator

import numpy as np


class LoopRateTimer:
    def __init__(self, framerate):
        self.set_framerate(framerate)
        self.start_time = 0

    def regulate(self):
        """
        Regulates the framerate of the timer.

        First Iteration: sets the start time.
        Subsequent iterations: sleeps for the remaining time of the loop time.
        """

        if self.start_time == 0:
            self.start_time = time.time()
            return

        elapsed = time.time() - self.start_time
        if elapsed < self.loop_time:
            time.sleep(self.loop_time - elapsed)
        self.start_time = time.time()

    def set_framerate(self, framerate):
        self.framerate = framerate
        self.loop_time = 1 / framerate


class PerformanceTimer(ContextDecorator):
    def __init__(self, history_length=1000):
        """Generates a PerformanceTimer object.

        Args:
            history_length (int, optional): Length of time_history.
                Defaults to 1000.

        """
        self.time_history = []
        self.history_length = history_length

    def start(self):
        self.start_time = time.time()

    def stop(self):
        elapsed = time.time() - self.start_time

        if len(self.time_history) > self.history_length:
            self.time_history.pop(0)

        self.time_history.append(elapsed)

    def get_average(self):
        return np.mean(self.time_history)

    def get_std_deviation(self):
        return np.std(self.time_history)

    def is_initialized(self):
        """
        Track for two iterations. If initialized, we can rely on the
        statistics.

        Returns:
            bool: _description_
        """
        return len(self.time_history) >= 2

    def __str__(self) -> str:
        num_digits = 4
        str = (
            f"Average: {np.round(self.get_average(), num_digits)}s, "
            f"Std. Deviation: {np.round(self.get_std_deviation(), num_digits)}s"
        )
        return str

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
