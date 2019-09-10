import os

DIR = os.path.dirname(os.path.abspath(__file__))
HAL_model_file = os.path.join(DIR, "HAL-9000.h5")
HAL_responses_file = os.path.join(DIR, "HAL_9000_responses.json")
MAG_FILE = os.path.join(DIR, "glove.840B.300d.magnitude")
DATA_FILE = os.path.join(DIR, "intents.csv")

MAX_SEQ_LEN = 28
batch_size = 32
