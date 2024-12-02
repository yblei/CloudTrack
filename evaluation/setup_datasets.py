import wget
import zipfile
import os

DATASET_BASE_DIR = "datasets"


def download_and_unzip(url, dest_folder):
    # check if dest folder exists. If yes, return
    if os.path.exists(dest_folder):
        print(f"Folder {dest_folder} exists. Skipping download.")
        return

    # make message
    print(f"Downloading and unzipping {url} to {dest_folder}.")

    # Download the zip file
    zip_file = wget.download(url)
    # Unzip the file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_folder)

    # remove zip file
    os.remove(zip_file)


def main():
    # setup davis 2017 trainval dataset
    url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    dest_folder = os.path.join(DATASET_BASE_DIR, "davis_trainval")
    download_and_unzip(url, dest_folder)

    # setup davis 2017 test-dev dataset
    url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip"
    dest_folder = os.path.join(DATASET_BASE_DIR, "davis_test_dev")
    download_and_unzip(url, dest_folder)

    # setup davis 2017 test-challenge dataset
    url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip"
    dest_folder = os.path.join(DATASET_BASE_DIR, "davis_test_challenge")
    download_and_unzip(url, dest_folder)

    # setup refering davis
    url = "https://www.mpi-inf.mpg.de/fileadmin/inf/d2/khoreva/davis_text_annotations.zip"
    dest_folder = os.path.join(DATASET_BASE_DIR, "refering_davis_annotations")
    download_and_unzip(url, dest_folder)

    # setup kitti dataset
    url = "https://www.cvlibs.net/datasets/kitti/user_login.php"

    # tao dataset
    url = "https://motchallenge.net/data/1-TAO_TRAIN.zip" 
    dest_folder = os.path.join(DATASET_BASE_DIR, "tao_train")
    download_and_unzip(url, dest_folder)

    url = "https://motchallenge.net/data/3-TAO_TEST.zip"
    dest_folder = os.path.join(DATASET_BASE_DIR, "tao_test")
    download_and_unzip(url, dest_folder)

    # run vot initialize vot2022/stb in console
    

    #wget "https://motchallenge.net/data/1-TAO_TRAIN.zip"
    #wget "https://motchallenge.net/data/2-TAO_VAL.zip" 
    #wget "https://motchallenge.net/data/3-TAO_TEST.zip" 

    

if __name__ == "__main__":
    main()
