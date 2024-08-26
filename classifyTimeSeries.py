
# from Blackbox_classifier_FCN.LITE.predict import predict_lite
from KerasModels.load_keras_model import model_classify


def classify(model_name, time_series):
    class_pred = model_classify(model_name=model_name, time_series=time_series)
    return int(class_pred)
