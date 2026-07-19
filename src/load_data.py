import numpy as np
import idx2numpy
import os
import kagglehub

def get_dataset():
    
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    path = kagglehub.dataset_download("hojjatk/mnist-dataset")

    train_images = idx2numpy.convert_from_file(os.path.join(path, "train-images.idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(os.path.join(path, "train-labels.idx1-ubyte"))

    test_images = idx2numpy.convert_from_file(os.path.join(path, "t10k-images.idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(os.path.join(path, "t10k-labels.idx1-ubyte"))
    
    return train_images, train_labels, test_images, test_labels
