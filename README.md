# Forecasting Epidemic Spread with Recurrent Graph Gate Fusion Transformers

This repository contains the code for the paper "Forecasting Epidemic Spread with Recurrent Graph Gate Fusion Transformers" submitted to the IEEE Journal of Biomedical and Health Informatics.

## Authors

- **Minkyoung Kim**
  - **Email:** minky@yonsei.ac.kr
  - **Affiliation:** Yonsei University
- **Jae Heon Kim**
  - **Email:** jhk774@yonsei.ac.kr
  - **Affiliation:** Yonsei University
- **Beakcheol Jang (Corresponding Author)**
  - **Email:** bjang@yonsei.ac.kr
  - **Affiliation:** Yonsei University

## Data
Before running the training and testing, please download the data folder from [Code Ocean Capsule](https://codeocean.com/capsule/0157271/tree) and add it to this repository.


## Installation

Ensure you have Python 3.9 installed. Setup your environment and install the necessary libraries using the following commands:

```bash
conda create -n regraft39 python=3.9
conda activate regraft39
pip install -r requirements.txt
```

## Training

To train and test the model, run:
```
python main.py
```
Outputs will be saved in the results folder.

## Citation

If you use this code in your research, please cite:
```
"Code for Forecasting Epidemic Spread with Recurrent Graph Gate Fusion Transformers," https://github.com/mfriendly/ReGraFT/
```

The implementation of our model is based on the code from the following repositories:
- [https://github.com/EternityZY/STNSCM](https://github.com/EternityZY/STNSCM)
- [https://github.com/mattsherar/Temporal_Fusion_Transform](https://github.com/mattsherar/Temporal_Fusion_Transform)

