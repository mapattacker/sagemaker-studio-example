"""refactored from https://www.kaggle.com/prajaktaparate14/audio-classification"""

import os

import joblib
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm



class preprocess:

    def mfcc_extractor(self, file):
        """convert audio file into mfcc features"""
        audio, sample_rate = librosa.load(file, res_type="kaiser_fast")
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features


    def f_extractor(self, df, idx):
        """some processing before feature extraction"""
        row = df.iloc[idx]
        # sagemaker does not allow subdirectories
        # file_name = os.path.join(self.audio_dataset_path, f'fold{row["fold"]}', row["slice_file_name"])
        file_name = os.path.join(self.audio_dataset_path, row["slice_file_name"])
        final_class_labels = row["class"]
        data = self.mfcc_extractor(file_name)
        return [data, final_class_labels]


    def encoder(self, y):
        """encode string labels to one-hot & save encode mapping"""
        # encode label string to int
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
        joblib.dump(labelencoder, "encoder.jb")
        # one-hot encoding
        y = to_categorical(y)
        return y


    def pipeline(self, audio_dataset_path, metadata_path, n_jobs=4, folds=None):
        """preprocess pipeline
        urbansounds metadata consists of the following columns
        ["slice_file_name","fsID","start","end","salience","fold","classID","class"]"""

        # load metadata
        self.audio_dataset_path = audio_dataset_path
        metadata = pd.read_csv(metadata_path)
        if folds:
            metadata = metadata[metadata["fold"].isin(folds)]

        # parallel extractor
        extracted_features = Parallel(n_jobs=n_jobs)(
            delayed(self.f_extractor)(metadata, idx) for idx in tqdm(range(len(metadata))))

        ## Converting extracted_features to pandas dataframe
        extracted_features_df = pd.DataFrame(extracted_features, columns=["feature","class"])

        X = np.array(extracted_features_df["feature"].tolist())
        y = np.array(extracted_features_df["class"].tolist())

        y = self.encoder(y)

        ## Train Test Split
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    audio_dataset_path = "UrbanSound8K/audio/"
    metadata_path = "UrbanSound8K.csv"
    P = preprocess()
    X_train, X_test, y_train, y_test = P.pipeline(audio_dataset_path, metadata_path, folds=[1])
