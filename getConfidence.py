# from Blackbox_classifier_FCN.LITE.predict import predict_lite
from KerasModels.load_keras_model import model_confidence
import numpy as np


def get_confidence(time_series: np.ndarray, data_set: str) -> float:
    confidence = None
    try:
        confidence = model_confidence(data_set, time_series)
    except Exception as e:
        print(e)
        print("Error!!!!")
        confidence = 0

    return confidence
