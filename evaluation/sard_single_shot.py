from omegaconf import DictConfig
import hydra
import glob
from pathlib import Path
from PIL import Image
from cloud_track.foundation_model_wrappers import DetectorVlmPipeline, GPTFourWrapper, GroundingDinoHuggingfaceWrapper, PaligemmaWrapper, LlavaWrapper, GLEEWrapper, GroundedSamWrapper
from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import system_prompt_from_description
from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import get_vlm_pipeline
import xmltodict
import json
from loguru import logger
import time
from tqdm import tqdm
try:
    from utils.fix_logging import fix_logging, handle_exception
except ImportError:
    logger.info("Could not import fix_logging from utils.fix_logging.")
import sys


def get_idxs(path):
    # find all .jpg files in the directory
    all_files = glob.glob(f"{str(path)}/*.jpg", recursive=True)
    all_files.sort()

    idxs = []
    for file in all_files:
        # get idx of file
        file = Path(file)
        idx = int(file.stem.replace("gss", ""))
        idxs.append(idx)

    idxs.sort()

    return idxs


def load_image_and_annotation(file_path, xml_path):
    # load s.png image as PIL from the current directory
    image = Image.open(file_path)

    # load xml
    with open(xml_path, "r") as f:
        xml = f.read()
        xml = xmltodict.parse(xml)
        annotation = xml["annotation"]["object"]

        # if only one object is present, the object is not a list
        if not isinstance(annotation, list):
            annotation = [annotation]

    return image, annotation


def get_prompts(config_name):
    if config_name == "yannik":
        # my prompts
        category = "person"
        system_description = "You are a drone on a search and rescue mission."
        prompt = "You are looking for a person, who is missing after being injured. Is this person in the image?"
    elif config_name == "reihaneh":
        # reihanes recommendations to make it better
        category = "person"
        system_description = "You are an intelligent AI assistant that helps a drone in a search and rescue mission."
        prompt = "The drone needs to find an injured person. Is this person in the image?"

    elif config_name == "reihaneh_2":
        # reihanes recommendations to make it better
        category = "person"
        system_description = "You are an intelligent AI assistant that helps a drone in a search and rescue mission."
        prompt = "The drone needs to find an injured person. Is this person in this image injured? Start your justification with 'Lets analyze the image'."

    return system_description, prompt, category


@hydra.main(config_path="../conf", config_name="sard_single_shot_evaluation")
@logger.catch  # So kriegen wir die Exceptions in die Logdatei!!!
def main(cfg: DictConfig):
    """
    Running the evaluation on the SARD dataset in a single shot manner. We parse 
    every image through the DetectorVlmPipeline. We then ask, if the object is 
    hurt or injured. we treat the classes "seated and "laying_down" as "injured".
    All others are treated as "not injured".

    Args:
        cfg (DictConfig): _description_
    """
    fix_logging()

    # paths
    hydra_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    sard_dir = Path("/home/blei/flextrack_sol/datasets/SARD-fixed/SARD")

    # config
    save_every = 10
    benchmark = True
    experiment_name = cfg.experiment  # could be sar_shirt, sar_injury or sar_person
    vl_model_name = cfg.vlm.name
    detector_model = cfg.detector

    # log vl_model_name and exp
    logger.info(
        f"Running {experiment_name} with {vl_model_name} and {detector_model}")

    # get prompts from config
    category = "person"  # war mal person. human ...
    exp_name_key = f"{experiment_name}_prompt"
    system_description = cfg.vlm.prompt_set["system_prompt"]
    prompt = cfg.vlm.prompt_set[exp_name_key]

    # auto config
    simulate_time_delay = False
    render_images = True
    frame_limit = 5
    if benchmark:
        simulate_time_delay = True
        render_images = False
        frame_limit = -1

    # create model
    model = get_vlm_pipeline(vl_model_name, system_description,
                      simulate_time_delay, detector_model)

    logger.info(f"Logging to {hydra_dir}")

    # make image dir, delete if exists
    image_dir = hydra_dir / "images"
    if image_dir.exists():
        for file in image_dir.glob("*"):
            file.unlink()
    image_dir.mkdir()

    result_dict = {}
    result_dict["metadata"] = {
        "category": category,
        "system_description": system_description,
        "user": prompt,
        "vl_model": vl_model_name,
        "detector": detector_model
    }

    idxs = get_idxs(sard_dir)

    # delete
    skip = False
    ##
    count = 0
    for idx in tqdm(idxs):
        if int(idx) == 1750:
            skip = False
        if skip:
            continue
        name = f"gss{idx}"
        file_path = sard_dir / f"{name}.jpg"
        xml_path = sard_dir / f"{name}.xml"

        if not xml_path.exists():
            logger.warning(f"Skipping {name} due to missing gt file.")
            continue

        image, annotations = load_image_and_annotation(file_path, xml_path)

        # run inference
        start = time.time()
        image_pil, masks, boxes_filt, scores, matches, labels = model.run_inference(
            category=category,
            description=prompt,
            image=image,
            filter_results=False,
            mark_results=render_images
        )
        end = time.time()

        results = []
        for box, match, scores, label in zip(boxes_filt, matches, scores, labels):
            results.append({
                "box": box.tolist(),
                "score": scores,
                "match": match,  # if true: cathegory: yes - gpt: yes. ELSE: cathegory: yes - gpt: no
                "label": label
            })

        result_dict[name] = {
            "image": f"{name}.jpg",
            "result": results,
            "annotation": annotations,
            "time_s": start - end,
        }

        # save image
        if render_images:
            image_pil.save(image_dir / f"{name}.jpg")

        if idx % save_every == 0:
            with open(hydra_dir / "sard_single_shot.json", "w") as f:
                json.dump(result_dict, f, indent=4)

        count += 1
        if count == frame_limit:
            logger.warning(f"Terminated due to frame limit.")
            break

    logger.info("Done.")

if __name__ == "__main__":
    main()
