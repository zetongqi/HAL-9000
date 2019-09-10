# HAL-9000: Conversational Agent
## Hello Dave

![](HAL.png)

to activate the virtual envionment: pipenv shell

to install all the libraries required to run HAL-9000: pipenv install --skip-lock

HAL-9000 uses a pretrained 300D GloVe embedding: https://github.com/plasticityai/magnitude

to downlaod the embedding, run: python3 mag_model.py

to train your own HAL-9000, change the settings in HAL_training.py and run: python3 HAL_training.py

to have a conversation with HAL-9000, run: python3 HAL-9000.py
