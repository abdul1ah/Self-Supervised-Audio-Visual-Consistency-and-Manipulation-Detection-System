# Self-Supervised Audio-Visual Consistency & Manipulation Detection

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)

Multimodal deep learning system to detect audio-visual mismatches using
a dual-stream architecture.

------------------------------------------------------------------------

## Architecture

-   **Visual (3D ResNet-18):** Learns temporal facial dynamics from
    video clips
-   **Audio (2D ResNet-18):** Processes log Mel-spectrograms
-   **Fusion:** Concatenation → FC layers → authenticity score

------------------------------------------------------------------------

## Project Structure

``` text
.
├── backend/              # Inference Logic (face detection + audio handling)
├── checkpoints/          # Saved model weights
├── dataset/              # Dataset Generation Scripts
├── demo/                 # Sample videos for testing
├── my_model_results/     # AUC / Confusion Matrix
├── notebooks/            # Training Notebook
|
|── src/                  # Core training code
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── models.py
│   ├── preprocess.py
│   └── train.py
|
|
├── requirements          # Dependencies
└── README.md
```

------------------------------------------------------------------------

## Setup

``` bash
git clone https://github.com/abdul1ah/Self-Supervised-Audio-Visual-Consistency-and-Manipulation-Detection-System.git
cd Self-Supervised-Audio-Visual-Consistency-and-Manipulation-Detection-System
pip install -r requirements
```

------------------------------------------------------------------------

## Inference

``` bash
python backend/inference_pipeline.py
```

-   Face detection via MTCNN
-   Audio resampled to 48kHz
-   Supports variable input formats

------------------------------------------------------------------------

## Training

-   Dataset: RAVDESS
-   Strategy: **80% intra-actor hard-negative sampling**
-   Learns audio-visual synchronization (not identity)

------------------------------------------------------------------------

## Limitations

-   Sensitive to domain shift
-   Fails on:
    -   Variable frame rate (VFR)
    -   Noisy / compressed audio

------------------------------------------------------------------------

## Future Work

-   Train on VoxCeleb / DFDC
-   Improve robustness to real-world data
-   Handle VFR more reliably
