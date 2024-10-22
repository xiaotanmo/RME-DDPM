# RME-DDPM: Conditional Diffusion Models for Radio Map Estimation

## Overview

**RME-DDPM** is a novel approach for radio map estimation using conditional diffusion probabilistic models. Radio map estimation is a critical task in telecommunications, enabling the prediction of signal strength across various environments. RME-DDPM integrates diffusion processes with environmental context (e.g., building layouts, transmitter locations) to generate accurate radio maps. This model significantly improves upon traditional methods, providing more efficient, scalable, and accurate predictions for signal strength distribution in both urban and rural settings.

This repository contains the code implementation for RME-DDPM, including training, testing, and evaluation scripts.

## Usage

To run the code, follow these steps:

1. **Clone the respository**:
```
git clone https://github.com/xiaotanmo/RME-DDPM.git
cd RME-DDPM
```

2. **Set up the environment**:
```
pip install -r requirements.txt
```
   
3. **Dataset Preparation**:
RadioMapSeer(https://radiomapseer.github.io/)

4. **Training/Resume Training**:
Set the path:
```
"path": {
	"resume_state": "PATH/TO/CKPT"
},
```

Run the script:
```python
python run.py -p train -c config/radiomap.json
```

5. **Testing**:
Run the script:
```python
python run.py -p test -c config/inpainting_celebahq.json
```

## Acknowledge
## Acknowledgement

This work, **RME-DDPM: Conditional Diffusion Models for Radio Map Estimation**, is based on the code from the [Palette Image-to-Image Diffusion Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/tree/main) repository. I am grateful to the contributors for their work, which has been a helpful resource in developing this project.
