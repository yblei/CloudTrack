import cv2

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
            print(f"Supported: {width}x{height}")
        else:
            print(f"Not supported: {width}x{height} (Got {int(actual_width)}x{int(actual_height)})")

    cap.release()
    return supported

if __name__ == "__main__":
    test_resolutions()
