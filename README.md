# Infrared Face Recognization

User authentication with thermal/infrared sensor arrays. This repo is the implementation of course project for Computational Vison (22 fall) at Peking University, supervised by Prof. Shanghang Zhang. 
Group members: Bo Liang, Xinming Wei and Chucai Wang.

## Getting Started

### Dataset
You can find our dataset at [dataset link](https://disk.pku.edu.cn:443/link/5986AA7536C91C91935EE3273B58B548).

### Dataset Preparation
Put the thermal images (.png) and corresponding labels (.xml) under `dataset/figure` and `dataset/label` respectively. The image/label pair should be named like 3.45.png/xml, where ‘3’ denotes the category and ‘45’ denotes the index.

### Customize configurations
Copy the sample config with `cp config.sample.py config.py`, then you can modify the settings (e.g., output_dir, batchsize, etc.) in `config.py`

## Run

### Training
```shell
python main.py prepare
python main.py train
```

### Test
```shell
# Evaluate specified images and return annotated ones.
python main.py eval
# Evaluate on the test set and return acc. metrics (confuse matrix, overall acc.)
python main.py test
```

