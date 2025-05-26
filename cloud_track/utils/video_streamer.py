from pathlib import Path
from loguru import logger
import copy
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

        self.software_resize = copy.deepcopy(resize)  # enable resizing by default
        self.is_network_stream = False
        
        if isinstance(source, int) or source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
            # set resolution
            if resize:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resize[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resize[1])

                # Certain Cameras only support certain resolutions and select the closest one.
                # So we need to check if the camera supports the resolution we want.
                # check if the camera supports the resolution
                # pull test frame to get the camera resolution
                success, test_image = self.cap.read()
                if not success:
                    raise RuntimeError("Could not read from camera.")
                
                camera_width, camera_height = (test_image.shape[1], test_image.shape[0])
                
                if camera_width == self.resize[0] and camera_height == self.resize[1]:
                    self.software_resize = False
                else:
                    self.cap.release()
                    logger.warning(
                        f"Warning: Camera does not support resolution {self.resize}. "
                        f"Using software resizing instead. This might skew the image."
                    )
                    logger.info(
                        f"Supported resolutions are: {test_resolutions(int(source))}"
                    )
                    self.cap = cv2.VideoCapture(int(source))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resize[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resize[1])
                
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(source, str):
            if "rtmp://" in source:
                # if the source is a rtmp stream, use the rtmp url
                self.cap = cv2.VideoCapture(source) 
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.is_network_stream = True
            else:           
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
        
        if self.is_network_stream:
            # this helps to reduce latency
            for _ in range(5):
                success, image = self.cap.read()

        if not success:
            raise StopIteration

        if self.software_resize:
            assert isinstance(self.software_resize, tuple)
            image = cv2.resize(
                image, self.software_resize, interpolation=cv2.INTER_AREA
            )

        if self.color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.frame_count += 1
        return image
    
def test_resolutions(camera_index=0):
    common_resolutions = [
        (160, 120),    # QQVGA
        (320, 240),    # QVGA
        (640, 480),    # VGA
        (800, 600),    # SVGA
        (1024, 768),   # XGA
        (1280, 720),   # HD
        (1280, 1024),  # SXGA
        (1600, 1200),  # UXGA
        (1920, 1080),  # Full HD
        (2048, 1536),  # QXGA
        (2592, 1944),  # 5MP
        (3840, 2160),  # 4K
    ]

    cap = cv2.VideoCapture(camera_index)
    supported = []

    for width, height in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if int(actual_width) == width and int(actual_height) == height:
            supported.append((width, height))
            logger.info(f"Supported: {width}x{height}")
        else:
            logger.info(f"Not supported: {width}x{height} (Got {int(actual_width)}x{int(actual_height)})")
            pass

    cap.release()
    return supported


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
