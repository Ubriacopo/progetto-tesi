import json
import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
import requests
import zipfile_inflate64 as zipfile
from dotenv import load_dotenv
from requests.auth import HTTPDigestAuth

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


# todo doc once done
def download(authentication: HTTPDigestAuth, base_path: str, output_path: str, filename: str, destination_folder: str):
    """

    :param destination_folder:
    :param authentication:
    :param base_path:
    :param output_path:
    :param filename:
    :return:
    """
    logging.info("Working on " + filename)
    response = requests.get(f"{base_path}{filename}", auth=authentication)

    if response.status_code != 200:
        logging.info(response.status_code)
        logging.warning("File not found or something went wrong!")
        return

    file_path = f"{output_path}{filename}" if output_path.endswith("/") else f"{output_path}/{filename}"
    with open(file_path, "wb") as file:
        [file.write(chunk) for chunk in response]

    zipfile.ZipFile(file_path, 'r').extractall(path=f"{out_path}/{destination_folder}")
    os.remove(file_path)
    if Path(output_path + "/" + filename.split(".")[0]).is_dir():
        move_to_root_folder(output_path, output_path + "/" + filename.split(".")[0])

    logging.info("Downloaded file: {0}".format(file_path))


def move_to_root_folder(root_path, cur_path):
    # Taken from here:https://stackoverflow.com/questions/8428954/move-child-folder-contents-to-parent-folder-in-python
    for filename in os.listdir(cur_path):

        if os.path.isfile(os.path.join(cur_path, filename)):
            shutil.move(os.path.join(cur_path, filename), os.path.join(root_path, filename))

        elif os.path.isdir(os.path.join(cur_path, filename)):
            move_to_root_folder(root_pathz, os.path.join(cur_path, filename))

    # remove empty folders
    if cur_path != root_path:
        os.rmdir(cur_path)


def download_amigos_resource(filename: str, authentication: HTTPDigestAuth, base_path: str,
                             output_path: str, destination_folder: str,
                             seen_file_names: list[str], seen_file_names_path: str,
                             failed_files: list[dict], failed_files_path: str):
    if not filename in seen_file_names:
        try:
            download(authentication, base_path, output_path, filename, destination_folder)
            # Store the seen configuration
        except Exception as e:
            failed_files.append({"filename": filename, "error": str(e)})
            json.dump(failed_files, open(failed_files_path, 'w'))

        seen_file_names.append(filename)
        json.dump(seen_file_names, open(seen_file_names_path, 'w'))

def download_amigos_listing(authentication: HTTPDigestAuth, base_path: str,
                            output_path: str, file_template_name: str, destination_folder: str,
                            seen_file_names: list[str], seen_file_names_path: str, failed_files: list[dict],
                            failed_files_path: str, k: int = 40, two_vars: bool = False):
    """

    :param seen_file_names_path:
    :param seen_file_names:
    :param destination_folder:
    :param authentication:
    :param base_path:
    :param output_path:
    :param file_template_name:
    :param k:
    :param two_vars:
    :return:
    """
    # Download all files.
    for i in range(k):
        if two_vars:
            for j in range(k):
                filename = file_template_name.format(str(i + 1).zfill(2), str(j + 1).zfill(2))
                download_amigos_resource(filename, authentication, base_path, output_path, destination_folder,
                                         seen_file_names, seen_file_names_path, failed_files, failed_files_path)
        else:
            filename = file_template_name.format(str(i + 1).zfill(2))
            download_amigos_resource(filename, authentication, base_path, output_path, destination_folder,
                                     seen_file_names, seen_file_names_path, failed_files, failed_files_path)


if __name__ == "__main__":
    load_dotenv("../.env")
    auth = HTTPDigestAuth(os.getenv("AMIGOS_USERNAME"), os.getenv("AMIGOS_PASSWORD"))
    amigos_base_path = "https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/"

    # Read arguments of run
    parser = ArgumentParser()
    # Arg to get the destination path of the processing
    parser.add_argument("-p", "--path", dest="output_path", help="Destination path", default="./")
    args = parser.parse_args()

    seen_files: list[str] = []
    # Ripristina sessione precedente di download se fatta
    seen_files_path = "./seen-files.tmp.json"
    if Path(seen_files_path).is_file():
        seen_files += json.load(open(seen_files_path))
        logging.info(f"Restored seen files list. Already tried to download {len(seen_files)} files")

    failed_files: list = []  # List of dictionary with filename + if downloaded
    failed_files_path = "./failed-files.tmp.json"
    if Path(failed_files_path).is_file():
        failed_files += json.load(open(failed_files_path))
        logging.info(f"Restored failed files list. {len(failed_files)} files failed to process")

    out_path = args.output_path
    logging.info("Resources will be stored in: {0}".format(out_path))

    metadata = "Metadata_ods.zip"
    # Download the resource
    if not metadata in seen_files:
        # Create folder if not existing
        Path(f"{out_path}/metadata").mkdir(exist_ok=True)
        download(auth, amigos_base_path, f"{out_path}/metadata", filename=metadata, destination_folder="metadata")

        # Set the downloaded file as seen
        seen_files.append(metadata)
        json.dump(seen_files, open(seen_files_path, 'w'))

    # Typo da parte di Amigos :)
    for annotation in ['SelfAsessment_ods.zip', "External_Annotations_ods.zip"]:
        if not annotation in seen_files:
            Path(f"{out_path}/annotation").mkdir(exist_ok=True)
            download(auth, amigos_base_path, output_path=f"{out_path}/annotation",
                     filename=annotation, destination_folder="annotation")

            seen_files.append(annotation)
            json.dump(seen_files, open(seen_files_path, 'w'))

    pre_processed = "Data_Preprocessed_P{0}.zip"
    Path(f"{out_path}/pre_processed").mkdir(exist_ok=True)
    download_amigos_listing(
        auth, base_path=amigos_base_path, output_path=f"{out_path}/pre_processed", file_template_name=pre_processed,
        destination_folder="pre_processed", seen_file_names=seen_files, seen_file_names_path=seen_files_path,
        failed_files=failed_files, failed_files_path=failed_files_path
    )

    # Experiment divided
    base_exp1 = "Exp1_P{{0}}_{experiment}.zip"
    # Most of these will fail, but I think it's the fastest alternative
    individual_exp2 = "Exp2_L{{0}}_Indiv_N{{1}}_{experiment}.zip"
    group_exp2 = "Exp2_L{{0}}_Group_N{{1}}_{experiment}.zip"

    for experiment in ['timestamps', 'face', 'rgb', 'depth']:
        out = f"{out_path}/{experiment}"
        Path(out).mkdir(exist_ok=True)

        exp1 = base_exp1.format(experiment=experiment)
        download_amigos_listing(
            auth, base_path=amigos_base_path, output_path=out, file_template_name=exp1,
            destination_folder=experiment,
            seen_file_names=seen_files, seen_file_names_path=seen_files_path,
            failed_files=failed_files, failed_files_path=failed_files_path
        )

        i_exp2 = individual_exp2.format(experiment=experiment)
        download_amigos_listing(
            auth, base_path=amigos_base_path, output_path=out, file_template_name=i_exp2,
            two_vars=True, destination_folder=experiment,
            seen_file_names=seen_files, seen_file_names_path=seen_files_path,
            failed_files=failed_files, failed_files_path=failed_files_path
        )

        g_exp2 = group_exp2.format(experiment=experiment)
        download_amigos_listing(
            auth, base_path=amigos_base_path, output_path=out, file_template_name=g_exp2,
            two_vars=True, destination_folder=experiment,
            seen_file_names=seen_files, seen_file_names_path=seen_files_path,
            failed_files=failed_files, failed_files_path=failed_files_path
        )
