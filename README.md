# UAV RF Fingerprint Recognition
This repository contains the implementation of RF-TCNet, a lightweight deep learning model for RF fingerprint recognition of unmanned aerial vehicles (UAVs).
The model performs accurate classification of drone RF signals with extremely low trainable parameters through an efficient topology compression network specifically designed for RF spectrograms.

# Project Overview
This project focuses on end-to-end classification of drone RF signals.
The workflow consists of:

Pre-processing the raw RF data.

Generating time-frequency spectrograms using the ECSG method.

Training and evaluating a lightweight deep model (RF-TCNet) that combines the strengths of convolutional and transformer-based architectures.

# Datasets
Dataset description Public datasets DroneRF and DroneRFa.

# Installation
Clone this repository and install dependencies:
git clone https://github.com/yourusername/UAV-RF-TCNet.git
cd UAV-RF-TCNet
pip install -r requirements.txt

# Usage
Prepare spectrograms
Download the DroneRF and DroneRFa datasets.
Apply the ECSG preprocessing to generate spectrograms.
Split datasets
Use the provided dataset partitioning script to create train/validation/test splits.
Train the model
python RF_TCNet_Train.py
python RF_TCNet_Test.py

# Model Highlights
Topology Compression: Reduces redundant connections while retaining key spectral features.
Dynamic Frequency Attention: Adapts to variations in UAV transmission characteristics.
Edge-Efficient Design: Achieves low latency and small model size without sacrificing accuracy.

# License
This project is released under the MIT License.
See the LICENSE.

# Acknowledgements
We thank the authors of the DroneRF and DroneRFa datasets for making their data publicly available.
