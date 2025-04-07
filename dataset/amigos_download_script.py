import requests
from requests.auth import HTTPDigestAuth
import zipfile
import os

amigos_base_path = "https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/"
pre_processed_template_name = "Data_Preprocessed_P{}.zip"
# todo finish
# Download the file.
auth = HTTPDigestAuth(env.unsername, env.password)
pre_processed_url = "https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/data/Data_Preprocessed_P01.zip"

res = requests.get(pre_processed_url, auth=auth)

print(res)
if res.status_code == 404:
    print("File not found")

# Store the file
with open("Data_Preprocessed_P01.zip", "wb") as file:
    [file.write(chunk) for chunk in res]

# Unpack the file
zipfile.ZipFile("Data_Preprocessed_P01.zip", 'r').extractall()

# Remove the old file
os.remove("Data_Preprocessed_P01.zip")
