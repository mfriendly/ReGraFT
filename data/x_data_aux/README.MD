# Prediction of Epidemic Spread with Recurrent Graph Fusion Transformers

This repository contains the PyTorch implementation of the Recurrent Graph Fusion Transformer (ReGraFT) for predicting epidemic spread, as described in our recent publication submitted to IEEE JBHI. The model uses recurrent neural network and graph neural network architectures to forecast epidemic spread across regions based on historical data.
The implementation of our proposed model is based on the code from the \textit{STNSCM} repository  {https://github.com/EternityZY/STNSCM}.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed Python 3.7 or later.
*
* You have a PyTorch 1.7 or later installed.
* You have access to a machine with CUDA-compatible GPUs for model training.

## Installation
torch==2.3.0.dev20240216+cu118
torch-lr-finder==0.2.1
torch_geometric @ file:///usr/share/miniconda/envs/test/conda-bld/pyg_1697446357367/work
torchaudio==2.2.0.dev20240216+cu118
torcheval==0.0.7
torchmetrics==1.3.1
torchvision==0.18.0.dev20240216+cu118
1. Clone the repository:
   ```bash
   git clone https://github.com/your-github/regraft-epidemic-prediction.git
   cd regraft-epidemic-prediction
Install the required packages:

pip install -r requirements.txt
Usage
To train the model with the default parameters on the provided dataset, run the following command:

python Train_ReGraFT.py
Configuration
Modify the config.json file in the root directory to adjust training parameters such as batch size, learning rate, and number of epochs.

Testing
To test the model after training, run:


python Test_ReGraFT.py
This script will load the trained model and perform predictions on the test dataset.

Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request.

License
Distributed under the MIT License. See LICENSE for more information.



This README template provides a clear, concise guide for users to set up and run your model.


conda activate regraft39
pip install -f requirements.txt
python Train_ReGraFT.py