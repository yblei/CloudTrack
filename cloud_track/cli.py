# Put imports directly in the function - autocomplete becomes terribly slow otherwise
import sys

import typer
from click import Context
from loguru import logger
from typer.core import TyperGroup
from typing_extensions import Annotated

# Remove the default sink
logger.remove()

# Add a new sink with the desired log level
logger.add(
    sys.stderr, level="INFO"
)  # We only want to see warnings and errors in CLI mode


app = typer.Typer(
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)

# I found the emojis at emojipedia: https://emojipedia.org/
# The Sortcodes (i.e. :boom: can be found in technical Information-> Shortcodes
# when clicking on the icon)


def complete_vlm():
    return [
        "llava-hf/llava-1.5-13b-hf",
        "llava-hf/llava-1.5-7b-hf",
        "gpt-4o-mini",
        "gpt-4o",
    ]


def complete_detector():
    return [
        "sam_hq",
        "sam_lq",
    ]


@app.command(help="Run the backend server.")
def backend(
    ip: Annotated[
        str, typer.Option(help="The IP adress to bind to.")
    ] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="The port to use.")] = 3000,
    system_prompt: Annotated[
        str, typer.Option(help="The system prompt to use.")
    ] = "Does the description match the image?",
    detector_name: Annotated[
        str,
        typer.Option(
            help="The Name of the detector. Use autocomplete for options."
            "requires the manual installation of grounding dino and downloads of"
            " weights.",
            autocompletion=complete_detector,
        ),
    ] = "sam_hq",
    vlm: Annotated[
        str,
        typer.Option(
            help="The Name of the Vision Language model. Use autocomplete for"
            "options.",
            autocompletion=complete_vlm,
        ),
    ] = "llava-hf/llava-1.5-13b-hf",
):
    from cloud_track.api_functions.run_backend import run_backend

    run_backend(ip, port, system_prompt, detector_name, vlm)


@app.command(rich_help_panel="Demos")
def video_demo(
    ip: Annotated[
        str, typer.Option(help="The IP adress of the backend.")
    ] = "http://127.0.0.1",
    port: Annotated[int, typer.Option(help="The port to use.")] = 3000,
    frontend_tracker: Annotated[
        str, typer.Option(help="The frontend tracker to use.")
    ] = "nano",
    frontend_tracker_threshold: Annotated[
        float, typer.Option(help="The threshold for the frontend tracker.")
    ] = 0.75,
    video_source: Annotated[
        str,
        typer.Option(
            help="The opencv video source to use. (Could be folder or video file)"
        ),
    ] = "assets/faint.mp4",
    cathegory: Annotated[
        str, typer.Option(help="The cathegory of the object.")
    ] = "a person",
    description: Annotated[
        str, typer.Option(help="The description of the object.")
    ] = "Is the person unconcious?",
    output_file: Annotated[
        str, typer.Option(help="The file to write the results to.")
    ] = None,
):
    """
    Run the [blue]command line demo[/blue]. :boom:
    """
    from cloud_track.api_functions.video_demo import run_video_demo

    # resolution: tuple = (1296, 720)
    resolution: tuple = (640, 480)

    run_video_demo(
        video_source=video_source,
        resolution=resolution,
        cathegory=cathegory,
        description=description,
        frontend_tracker=frontend_tracker,
        frontend_tracker_threshold=frontend_tracker_threshold,
        backend_address=ip,
        backend_port=port,
        output_file=output_file,
    )


@app.command(rich_help_panel="Demos")
def live_demo(
    ip: Annotated[
        str, typer.Option(help="The IP adress of the backend.")
    ] = "http://127.0.0.1",
    port: Annotated[int, typer.Option(help="The port to use.")] = 3000,
    frontend_tracker: Annotated[
        str, typer.Option(help="The frontend tracker to use.")
    ] = "nano",
    frontend_tracker_threshold: Annotated[
        float, typer.Option(help="The threshold for the frontend tracker.")
    ] = 0.75,
    video_source: Annotated[
        str,
        typer.Option(
            help="The opencv video source to use. (Could be folder or video file)"
        ),
    ] = "0",
    cathegory: Annotated[
        str, typer.Option(help="The cathegory of the object.")
    ] = "a person",
    description: Annotated[
        str, typer.Option(help="The description of the object.")
    ] = "Is the person unconcious?",
):
    """
    Run the [blue]live demo[/blue] with webcam input. :camera:
    """
    from cloud_track.api_functions.live_demo import run_live_demo

    resolution: tuple = (640, 480)

    video_source = (
        int(video_source) if video_source.isdigit() else video_source
    )

    run_live_demo(
        video_source=video_source,
        tracker_resolution=resolution,
        cathegory=cathegory,
        description=description,
        frontend_tracker=frontend_tracker,
        frontend_tracker_threshold=frontend_tracker_threshold,
        backend_address=ip,
        backend_port=port,
    )


def main():
    app()


if __name__ == "__main__":
    main()
