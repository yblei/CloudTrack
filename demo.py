from cloud_track.api_functions.video_demo import run_video_demo


if __name__ == "__main__":
    video_source: str = "assets/faint.mp4"
    resolution: tuple = (640, 480)
    cathegory: str = "a person"
    description: str = "Is the person unconcious?"
    frontend_tracker: str = "nano"
    frontend_tracker_threshold: float = 0.75
    use_network_backend: bool = False
    backend_address: str = "http://127.0.0.1"
    backend_port: int = 3000

    run_video_demo(
        video_source=video_source,
        resolution=resolution,
        cathegory=cathegory,
        description=description,
        frontend_tracker=frontend_tracker,
        frontend_tracker_threshold=frontend_tracker_threshold,
        backend_address=backend_address,
        backend_port=backend_port,
        output_file=None
    )
