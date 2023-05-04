import requests
import os
import zipfile
import shutil

output_dir = "data/downloaded"
os.makedirs(output_dir, exist_ok=True)

URL = "https://calcofi.org/downloads/database/CalCOFI_Database_194903-202001_csv_22Sep2021.zip"
response = requests.get(URL)
open(output_dir + "/bottle.zip", "wb").write(response.content)

with zipfile.ZipFile(output_dir + "/bottle.zip", 'r') as zip_ref:
        zip_ref.extractall(output_dir)

os.remove(output_dir + "/bottle.zip")
os.rename(output_dir + "/CalCOFI_Database_194903-202001_csv_22Sep2021/194903-202001_Bottle.csv", output_dir + "/bottle.csv")
shutil.rmtree(output_dir + "/CalCOFI_Database_194903-202001_csv_22Sep2021")
