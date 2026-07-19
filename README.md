# MNIST Digit Classification with ANN

A fully connected neural network (ANN) trained on the MNIST handwritten digits dataset, with hyperparameter tuning via Keras Tuner's Hyperband search. The final model achieves 98.58% test accuracy.

## Overview

This project builds an image classifier for handwritten digits (0-9) using the raw MNIST dataset in its original IDX binary format. Instead of manually picking an architecture, a Hyperband search was run to find the best combination of layer depth, layer width, dropout, and learning rate.

## Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits, each 28x28 pixels.

- train-images.idx3-ubyte: 60,000 training images
- train-labels.idx1-ubyte: 60,000 training labels
- t10k-images.idx3-ubyte: 10,000 test images
- t10k-labels.idx1-ubyte: 10,000 test labels

Class distribution is approximately balanced across all 10 digits, so no class balancing techniques were needed.

## Approach

1. Loaded raw IDX format image and label files
2. Normalized pixel values to a 0-1 range
3. Built a tunable ANN architecture:
   - Flatten layer to convert 28x28 images into a 784 length vector
   - 1 to 5 Dense layers with tunable unit counts (32 to 512)
   - BatchNormalization and ReLU activation after each Dense layer
   - Optional Dropout after each layer, with tunable dropout rate
   - Final Dense layer with 10 units and softmax activation
4. Used Keras Tuner's Hyperband search to explore the hyperparameter space:
   - Number of layers
   - Units per layer
   - Dropout usage and rate per layer
   - Learning rate
5. Trained the best found architecture on the full training set with EarlyStopping and ReduceLROnPlateau callbacks
6. Evaluated on the held out test set and generated a confusion matrix
7. Saved the final trained model for reuse

## Results

- Final test accuracy: 0.98580002784729
- Final test loss: 0.09160809963941574

The confusion matrix shows that most misclassifications occur between visually similar digits, such as 7 and 2, or 9 and 4.

## Reproducibility

All random operations were seeded for reproducibility:

```python
seed = 42
tf.keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()
os.environ["PYTHONHASHSEED"] = str(seed)
```

## Tech Stack

- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy
- Matplotlib and Seaborn
- scikit-learn

## Project Structure

```
mnist_dataset_model/
├── images/
│   ├── confusion_matrix.png
│   ├── model_architecture.png
│   └── training_curves.png
├── models/
│   └── mnist_ann_model.keras
├── src/
│   ├── __init__.py
│   └── load_data.py
├── tuner_results/
│   └── mnist_ann_search/    (not included, regenerate by rerunning the search)
├── model.ipynb              (main notebook: data loading, tuning, training, evaluation)
├── requirements.txt
└── README.md
```


## Setup

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Either run the whole notebook or load the model

## Model Architecture

The best architecture found by the search:

- Number of layers: 3
- Units per layer: 368, 160, 384
- Dropout applied selectively per layer
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: sparse categorical crossentropy

See images/model_architecture.png for a visual diagram of the full model.

## Future Improvements

- Extend to a Convolutional Neural Network (CNN) to push past the accuracy ceiling of a fully connected ANN
- Add data augmentation to improve robustness to handwriting variation
- Improve preprocessing for real-world handwritten photos (lighting, cropping, stroke thickness normalization), since MNIST-trained models generalize poorly to photos out of the box
- Experiment with ensembling multiple tuned models
- Build a simple web app (e.g. Streamlit) for interactive digit prediction

## License
MIT LICENSEgit pull origin main
This project is open source and available for personal and educational use.