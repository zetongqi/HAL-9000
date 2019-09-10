import os
import urllib.request

MAG_DOWNLOAD_URL = (
    "http://magnitude.plasticity.ai/glove/medium/glove.840B.300d.magnitude"
)
DIR = os.path.dirname(os.path.abspath(__file__))
MAG_FILE = os.path.join(DIR, "glove.840B.300d.magnitude")

if __name__ == "__main__":
    if not os.path.exists(MAG_FILE):
        print("downloading magnitude model")
        urllib.request.urlretrieve(MAG_DOWNLOAD_URL, MAG_FILE)
