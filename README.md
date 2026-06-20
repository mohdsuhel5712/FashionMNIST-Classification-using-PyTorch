# FashionNet - Fashion MNIST Classifier using PyTorch

## Overview

FashionNet is a deep learning project developed using PyTorch to classify fashion products from the Fashion MNIST dataset. The model can identify 10 different categories of clothing items such as T-shirts, dresses, shoes, bags, and more.

## Project Structure

```text
FASHION_CLASSIFIER/
│
├── data/
│   └── Fashion MNIST dataset
│
├── model.py              # Neural Network architecture
├── train.py              # Model training script
├── evaluate.py           # Model evaluation script
├── predict.py            # Prediction on new samples
├── fashion_model.pth     # Trained model weights
├── requirements.txt      # Project dependencies
└── README.md
```

## Features

* Fashion image classification
* Built using PyTorch
* Model training and evaluation
* Save and load trained models
* Predict clothing categories on unseen images

## Dataset

This project uses the Fashion MNIST dataset, which contains 70,000 grayscale images belonging to 10 fashion categories.

### Classes

| Label | Category    |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

## Installation

```bash
git clone <repository-url>
cd FASHION_CLASSIFIER

pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Evaluation

```bash
python evaluate.py
```

## Prediction

```bash
python predict.py
```

## Results

The model successfully learns to classify fashion items from grayscale images and achieves strong performance on the Fashion MNIST test dataset.

## What I Learned

* Deep Learning Fundamentals
* Neural Networks
* Image Classification
* Model Training & Evaluation
* Saving and Loading PyTorch Models
* Data Preprocessing

## Author

**Mohd Suhail**

Aspiring AI/ML Engineer passionate about Deep Learning, Computer Vision, and PyTorch.
