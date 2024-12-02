# from cloud_track.foundation_model_wrappers import GroundedSamWrapper
from cloud_track.api_functions.run_backend import run_backend

if __name__ == "__main__":
    ip = "0.0.0.0"  # you dont need to put http here - only on the client side
    port = 3000
    system_prompt = "You are an intelligent AI helping an object detection system to find objects of different classes in images."
    detector_model = "sam_hq"
    vlm_model = "llava-hf/llava-1.5-13b-hf"
    #vlm_model = "gpt-4o-mini"

    run_backend(ip, port, system_prompt, detector_model, vlm_model)
