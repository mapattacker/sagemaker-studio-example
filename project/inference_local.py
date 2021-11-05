import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from preprocess import preprocess
import yaml

conf = open("config/conf.yaml", "r")
conf = yaml.safe_load(conf)

P = preprocess()
model = load_model(conf["deploy"]["model_dir"])
labelencoder = joblib.load("encoder.jb")


def predict(file):

    # get probability of each class, & convert to class name
    predicted = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predicted, axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)

    return prediction_class


if __name__ == "__main__":
    file = "101415-3-0-2.wav"
    # extract features
    mfccs_scaled_features = P.mfcc_extractor(file)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)
#     np.save('dog_bark.n````py', mfccs_scaled_features)
    file = np.load('dog_bark.npy')
    
    x = predict(file)
    print(x)