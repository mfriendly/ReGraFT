# Forecasting Epidemic Spread with Recurrent Graph Gate Fusion Transformers

- **Authors:** Minkyoung Kim, Jae Heon Kim, Beakcheol Jang
- IEEE Journal of Biomedical and Health Informatics (JBHI)

## Project Overview

This repository provides the code and data for the paper titled **"Forecasting Epidemic Spread with Recurrent Graph Gate Fusion Transformers"**, which has been submitted to the IEEE JBHI. The paper presents a novel approach using **Multigraph-Gated Recurrent Unit (MGRU)** architecture for spatiotemporal forecasting of COVID-19 spread.


## Code Base and References

- The code is based on the following repositories:
  - [STNSCM](https://github.com/EternityZY/STNSCM) by EternityZY
  - [Temporal Fusion Transformer](https://github.com/mattsherar/Temporal_Fusion_Transform) by Matt Sherar
- The variable importance code is adapted from [pytorch-forecasting](https://github.com/sktime/pytorch-forecasting).

## Installation and Running

### For Linux/macOS Users:

```bash
git clone https://github.com/mfriendly/ReGraFT.git
cd ReGraFT
cd code
conda create -n regraft39 python=3.9
conda activate regraft39
pip install -r requirements.txt
chmod +x ./run.sh
./run.sh
```

### For Windows Users:

```bash
git clone https://github.com/mfriendly/ReGraFT.git
cd ReGraFT
cd code
conda create -n regraft39 python=3.9
conda activate regraft39
pip install -r requirements.txt
python main_Pipeline.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
