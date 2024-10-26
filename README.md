# Forecasting Epidemic Spread with Recurrent Graph Gate Fusion Transformers

- **Authors:** Minkyoung Kim, Jae Heon Kim, Beakcheol Jang
- **Submitted to:** IEEE Journal of Biomedical and Health Informatics (JBHI)

## Project Overview

This repository provides the code and data for the paper titled **"Forecasting Epidemic Spread with Recurrent Graph Gate Fusion Transformers"**, which has been submitted to the IEEE JBHI. The paper presents a novel approach using **Multigraph-Gated GRU (MGRU)** architecture for spatiotemporal forecasting of COVID-19 spread.


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
./run.sh
```

### For Windows Users:

1. **Clone the repository**:
   Open the command prompt and run the following:

   ```bash
   git clone https://github.com/mfriendly/ReGraFT.git
   cd ReGraFT\code
   ```

2. **Create Python 3.9 environment (`regraft39`)**:
   If you use `conda`, create a Python 3.9 environment by running:

   ```bash
   conda create -n regraft39 python=3.9
   conda activate regraft39
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the code**:
   ```bash
   python main_Pipeline.py
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
