from tensorflow.keras.models import load_model
import os
import psutil
import numpy as np
from tensorflow.keras import backend as K
from KerasModels.load_keras_model import load_keras_model


def get_model_memory_usage(model):
    # Calculate the number of parameters
    trainable_count = np.sum([np.prod(v._shape)
                             for v in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(v._shape)
                                 for v in model.non_trainable_weights])

    # Calculate the total number of parameters
    total_params = trainable_count + non_trainable_count

    # Assuming float32 data type (4 bytes per parameter)
    total_memory = total_params * 4  # in bytes

    # Convert to megabytes
    total_memory_MB = total_memory / (1024 ** 2)

    return total_memory_MB


# Function to get memory usage in MB

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # in MB


# Print memory usage before loading the model
print(f"Memory usage before loading model: {get_memory_usage():.2f} MB")

# Load your model
my_keras_model = load_keras_model("Chinatown.keras")

# Print memory usage after loading the model
print(f"Memory usage after loading model: {get_memory_usage():.2f} MB")

# Example usage:
