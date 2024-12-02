from loguru import logger


class ReInitRequest:
    """
    Is raised, when the point tracker failed. After this, we re-initialize.
    """

    def __init__(
        self,
        reason,
        message="Point tracker failed",
        unprocessed_frames: list = [],  # a list of flex images
    ):
        self.message = f"{message}: {reason}"
        self.unprocessed_frames = unprocessed_frames

    def __str__(self) -> str:
        return self.message
