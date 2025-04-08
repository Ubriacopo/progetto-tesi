import os
import zipfile
from argparse import ArgumentParser
from pathlib import Path
import logging
import requests
from dotenv import load_dotenv
from requests.auth import HTTPDigestAuth
import os, shutil, sys


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

    zipfile.ZipFile(file_path, 'r').extractall(path=destination_folder)
    os.remove(file_path)

    move_to_root_folder(output_path, output_path + "/" + filename.split(".")[0])
    logging.info("Downloaded file: {0}".format(file_path))


def move_to_root_folder(root_path, cur_path):
    # Taken from here:https://stackoverflow.com/questions/8428954/move-child-folder-contents-to-parent-folder-in-python
    for filename in os.listdir(cur_path):

        if os.path.isfile(os.path.join(cur_path, filename)):
            shutil.move(os.path.join(cur_path, filename), os.path.join(root_path, filename))

        elif os.path.isdir(os.path.join(cur_path, filename)):
            move_to_root_folder(root_path, os.path.join(cur_path, filename))

    # remove empty folders
    if cur_path != root_path:
        os.rmdir(cur_path)


def download_amigos_listing(authentication: HTTPDigestAuth, base_path: str,
                            output_path: str, file_template_name: str, destination_folder: str,
                            k: int = 40, two_vars: bool = False):
    """

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
                download(authentication, base_path, output_path, filename)

        else:
            filename = file_template_name.format(str(i + 1).zfill(2))
            download(authentication, base_path, output_path, filename, destination_folder)


if __name__ == "__main__":
    load_dotenv("../.env")
    auth = HTTPDigestAuth(os.getenv("AMIGOS_USERNAME"), os.getenv("AMIGOS_PASSWORD"))

    parser = ArgumentParser()
    # Arg to get the destination path of the processing
    parser.add_argument("-p", "--path", dest="output_path", help="Destination path", default="./")
    args = parser.parse_args()

    out_path = args.output_path
    logging.info("Resources will be stored in: {0}".format(out_path))

    amigos_base_path = "https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/"

    # Data
    pre_processed = "Data_Preprocessed_P{0}.zip"
    Path(f"{out_path}/pre_processed").mkdir(exist_ok=True)
    download_amigos_listing(auth, amigos_base_path, output_path=f"{out_path}/pre_processed",
                            file_template_name=pre_processed, destination_folder="pre_processed")

    # Experiment divided
    base_exp1 = "Exp1_P{{0}}_{experiment}.zip"
    # Most of these will fail, but I think it's the fastest alternative
    individual_exp2 = "Exp2_L{{0}}_Indiv_N{{1}}_{experiment}.zip"
    group_exp2 = "Exp2_L{{0}}_Group_N{{1}}_{experiment}.zip"

    for experiment in ['timestamps', 'face', 'rgb', 'depth']:
        out = f"{out_path}/{experiment}"
        Path(out).mkdir(exist_ok=True)

        exp1 = base_exp1.format(experiment=experiment)
        download_amigos_listing(auth, amigos_base_path, output_path=out,
                                file_template_name=exp1, destination_folder=experiment)

        i_exp2 = individual_exp2.format(experiment=experiment)
        download_amigos_listing(auth, amigos_base_path, output_path=out,
                                file_template_name=i_exp2, two_vars=True, destination_folder=experiment)

        g_exp2 = group_exp2.format(experiment=experiment)
        download_amigos_listing(auth, amigos_base_path, output_path=out,
                                file_template_name=g_exp2, two_vars=True, destination_folder=experiment)
