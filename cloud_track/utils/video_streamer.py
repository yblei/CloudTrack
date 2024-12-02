from pathlib import Path

import cv2


class VideoStreamer:
    def __init__(
        self,
        source: str,
        resize=False,
        image_name_pattern="Image%8d.jpg",
        color_mode="bgr",
        skip: int = 1,
        max_frame: int = -1,
    ):
        self.source = source
        self.resize = resize
        self.image_glob = image_name_pattern
        self.color_mode = color_mode
        self.skip = skip
        self.max_frame = max_frame
        self.frame_count = 0

        if isinstance(source, int) or source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
            # set resolution
            if resize:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resize[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resize[1])
                self.resize = False  # disable resizing since it's already done
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(source, str):
            source = Path(source)
            # error if the source is not a file or directory
            if not source.exists():
                raise FileNotFoundError(f"Source {source} does not exist.")
            if source.is_dir():
                # check, if there are any images found with this regex
                image_name_pattern = source / image_name_pattern
                self.cap = cv2.VideoCapture(str(image_name_pattern))
            else:
                # assume it's a video file
                self.cap = cv2.VideoCapture(str(source))

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_frame > 0 and self.frame_count >= self.max_frame:
            raise StopIteration

        for _ in range(self.skip):
            success, image = self.cap.read()

        if not success:
            raise StopIteration

        if self.resize:
            assert isinstance(self.resize, tuple)
            image = cv2.resize(
                image, self.resize, interpolation=cv2.INTER_AREA
            )

        if self.color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.frame_count += 1
        return image


if __name__ == "__main__":
    stream = VideoStreamer(
        "/home/blei/flextrack_sol/datasets/book", image_name_pattern="%8d.jpg"
    )
    for frame in stream:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    stream.cap.release()
    cv2.destroyAllWindows()
