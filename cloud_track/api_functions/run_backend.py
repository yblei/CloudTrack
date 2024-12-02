from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import (
    get_vlm_pipeline,
)
from cloud_track.rpc_communication.rpc_server import FmBackend


def run_backend(ip, port, system_prompt, detector_name, vlm_model):
    model = get_vlm_pipeline(
        vlm_model,
        system_prompt,
        simulate_time_delay=True,
        detector_name=detector_name,
    )

    rpc_server = FmBackend(model)
    rpc_server.start(ip, port)
