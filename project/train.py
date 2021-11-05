"""refactored from https://www.kaggle.com/prajaktaparate14/audio-classification"""

import argparse
import os
from time import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam




def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # model directory
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()



class train:

    def __init__(self, input_shape, num_labels, 
                    model_path="./model", 
                    encoder_path="encoder/encoder.jb"):
        """
        Args:
            input_shape (int): number of features
            num_labels (int): number of classes
        """

        self.input_shape = input_shape
        self.num_labels = num_labels
        self.model_path = model_path
        self.encoder_path = encoder_path


    def model_arch(self):
        """define NN architecture in keras"""

        model = Sequential()

        ### first Layer
        model.add(Dense(1024, input_shape=(self.input_shape,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        ### second Layer
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        ### third Layer
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        ### final Layer
        model.add(Dense(self.num_labels))
        model.add(Activation('softmax'))
        model.compile(
                loss='categorical_crossentropy', 
                metrics=['accuracy'], 
                optimizer='adam')

        return model


    def start_train(self, X_train, X_test, y_train, y_test, epochs, batch_size):
        """start training"""

        model = self.model_arch()
        checkpointer = ModelCheckpoint(filepath=self.model_path, 
                            verbose=1, 
                            save_best_only=True)
        
        start = time()
        model.fit(X_train, y_train, 
                batch_size=batch_size, 
                epochs=epochs, 
                validation_data=(X_test, y_test), 
                callbacks=[checkpointer])

        elapsed_time = round((time()-start)/60, 2)
        print(f'Training completed in time: {elapsed_time} min')
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f'Accuracy: {test_accuracy[1]}')
        


if __name__ == "__main__":
    # model dir
    try:
        model_dir = os.environ['SM_MODEL_DIR']
    except:
        model_dir = "./model"
    version = "0000000"
    model_dir = os.path.join(model_dir, version)
    
    
    from preprocess import *
    import yaml
    
    # load hyperparameters
    conf = open("config/conf.yaml", "r")
    conf = yaml.safe_load(conf)
    param = conf["hyperparameters"]
    
    # audio_dataset_path = "UrbanSound8K/audio/"
    audio_dataset_path = "/opt/ml/input/data/training"
    metadata_file = "metadata/UrbanSound8K.csv"
    P = preprocess()
    X_train, X_test, y_train, y_test = P.pipeline(audio_dataset_path, metadata_file, n_jobs=10, folds=[1])
    
    input_shape=len(X_train[0])
    num_labels=y_train.shape[1]

    t = train(input_shape, num_labels, model_path=model_dir)
    t.start_train(X_train, X_test, y_train, y_test, param["epoch"] , param["batch_size"])

