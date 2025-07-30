
# Ternary Neural Network on MNIST

This project implements a Ternary Neural Network (TNN) using PyTorch for the MNIST digit classification task. It uses ternary weights { -1, 0, +1 } in both convolutional and linear layers to significantly reduce memory and computation requirements while retaining good accuracy.



## Features

- Custom TernaryLinear and TernaryConv2d layers

- BatchNorm and ReLU activation

- Lightweight TernaryCNN architecture

- TensorBoard logging for training and validation

- Ternary weight distribution visualization

- Compatible with CPU and GPU


##  📂 Folder Structure


```bash
TernaryMNIST/
├── data/
│   └── download.py               # MNIST DataLoader logic
├── models/
│   ├── ternary_cnn.py           # Ternary CNN architecture
│   └── ternary_layers.py        # TernaryConv2d and TernaryLinear
├── utils/
│   ├── train.py                 # Training script
│   ├── eval.py              # Evaluation logic
│   └── visualizer.py            # Plots and ternary weight distribution
├── checkpoints/                 # Saved model weights
├── logs/                        # TensorBoard logs and plots
├── main.py                      # Main training entry point
├── requirements.txt             # Python dependencies
└── README.md

```

## Installation



```bash
  git clone https://github.com/09-prince/TernaryNN-MNIST-.git
  cd TernaryNN-MNIST-
  pip install -r requirements.txt
```
    
## Usage

```bash
python3 main.py
```
By default, it trains for 10 epochs on the MNIST dataset and logs metrics to TensorBoard.

## Evaluate

```bash
python3 -m utils.eval
```

## Launch TensorBoard

```bash
tensorboard --logdir=logs/
```
## Authors

- Created by [09-prince](https://github.com/09-prince)
