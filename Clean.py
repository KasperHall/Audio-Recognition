# %% Imports
from ast import Mult
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import scipy.sparse as sparse 

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

#%% DataLoader
class DataLoader():

    resize = preprocessing.Resizing(64, 64)
    norm = preprocessing.Normalization()
    def __init__(self, n_training = 6400, n_test=800) -> None:
    
        data_dir = pathlib.Path('data/mini_speech_commands')
        if not data_dir.exists():
            tf.keras.utils.get_file(
                'mini_speech_commands.zip',
                origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract=True,
                cache_dir='.', cache_subdir='data')

        commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        self.commands = commands[commands != 'README.md']
        print('Commands:', self.commands)

        filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        print('Number of total examples:', num_samples)
        print('Example file tensor:', filenames[0])

        train_files = filenames[:6400]
        val_files = filenames[6400: 6400 + 800]
        test_files = filenames[-800:]

        print('Training set size', len(train_files))
        print('Validation set size', len(val_files))
        print('Test set size', len(test_files))     

        AUTOTUNE = tf.data.AUTOTUNE
        files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.train_set = waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE).take(n_training)

        files_ds = tf.data.Dataset.from_tensor_slices(val_files)
        waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.val_set = waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

        files_ds = tf.data.Dataset.from_tensor_slices(test_files)
        waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.test_set = waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE).take(n_test)


    @staticmethod
    def decode_audio(audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    @staticmethod
    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] 

    @staticmethod
    def get_waveform_and_label(file_path):
        label = DataLoader.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = DataLoader.decode_audio(audio_binary)
        return waveform, label

    @classmethod
    def get_spectrogram(cls, waveform):
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)

        spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.math.abs(spectrogram)
        
        spectrogram = tf.math.pow(spectrogram, 0.2)
        spectrogram = tf.expand_dims(spectrogram, -1)

        spectrogram = DataLoader.resize(spectrogram)
        spectrogram = DataLoader.norm(spectrogram)

        return spectrogram 

    def get_spectrogram_and_label_id(self, audio, label):
        spectrogram = DataLoader.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == self.commands)
        return spectrogram, label_id

    def visualize(self):
        for spectrogram, label_id in self.train_set.take(1):
            
            print('Label:', label_id)
            print('Spectrogram shape:', spectrogram.shape)

            fig, axes = plt.subplots(1, figsize=(12, 8))
            self.plot_spectrogram(spectrogram.numpy(), axes)
            axes.set_title('Spectrogram')
            plt.show()

    @staticmethod
    def plot_spectrogram(spectrogram, ax):
        # Convert to frequencies to log scale and transpose so that the time is
        # represented in the x-axis (columns).
        spectrogram = spectrogram[:,:,0]

        log_spec = np.log(spectrogram.T)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)





#%% SingleReadoutLayer
class SingleReadoutLayer():
    """
    Predicts the most likely output class from a vector of inputs through ridge regression
    """
    def __init__(self, n_classes, ridge_parameter = 0.1) -> None:
        self.n_classes = n_classes
        self.ridge_parameter = ridge_parameter

        self.design_matrix = []
        self.target_output = []

    def predict(self, x):
        x = x.flatten()
        return np.argmax(x @ self.output_weights)

    def test(self, test_set):
        correct = 0
        for x, target in test_set:
            if self.predict(x) == target:
                correct += 1
                
        return correct/len(test_set)

    def add_sample(self, sample, target_id):
        sample = sample.reshape((1, sample.size))
        
        target = np.zeros((1,self.n_classes))
        target[0,target_id] = 1

        self.design_matrix.append(sample)
        self.target_output.append(target)

    def finalize_training(self):
        self.design_matrix = np.concatenate(self.design_matrix, axis=0)
        self.target_output = np.concatenate(self.target_output, axis=0)

        size = len(self.design_matrix[0,:])
        self.output_weights = np.linalg.inv(self.design_matrix.T @ self.design_matrix + self.ridge_parameter*np.eye(size)) @ self.design_matrix.T @ self.target_output

    def train(self, data_set):
        for sample, target in data_set:
            self.add_sample(sample, target)
            
        self.finalize_training()

#%% MultiReadoutLayer
class MultiReadoutLayer():
    """
    Predicts the most likely output class from a series of vectors through ridge regression
    """

    def __init__(self, n_classes: int, n_steps: int,  ridge_parameter = 0.1) -> None:
        self.readout_layers = [SingleReadoutLayer(n_classes, ridge_parameter) for i in range(n_steps)]
        self.n_classes = n_classes
        self.ridge_parameter = ridge_parameter

    @classmethod
    def from_layers(cls, readout_layers, n_classes, n_steps, ridge_parameter=0.1):
        layer = MultiReadoutLayer(n_classes, n_steps=n_steps, ridge_parameter=ridge_parameter)
        layer.readout_layers = readout_layers
        return layer

    def predict(self, x, plot=False):

        result = np.array([x_t.T @ l.output_weights for l, x_t in zip(self.readout_layers, x)])
        #result[result < np.median(result)] = 0

        if plot:           
            plt.figure(figsize=(16,16))
            plt.imshow(result.T, cmap='hot', interpolation='nearest')
            plt.show()
        
        return np.argmax(np.mean(result, 0))
    
    def test(self, test_set, reservoir):
        correct = 0
        for x, target in test_set.as_numpy_iterator():

            if not reservoir is None:
                x = reservoir.predict(x)

            if self.predict(x) == target:
                correct += 1
                
        return correct/len(test_set)
    
    def train(self, training_set, reservoir = None):
        for i, (x, y) in enumerate(training_set.as_numpy_iterator()):

            if not reservoir is None:
                x = reservoir.predict(x)

            for layer, x_t in zip(self.readout_layers, x):
                layer.add_sample(x_t, y)            
            if i%500==1:
                print(f"{i}/{len(training_set)} samples completed.")

        print("Inverting matrices...")
        for i, layer in enumerate(self.readout_layers):
            layer.finalize_training()
            if i%10==1:
                print(f"{i}/{len(self.readout_layers)} layers completed.")
        
        print("Training completed!")

#%% Reservoir
class ReservoirLayer():
    """
    Simple Echo state network layer
    """
    def __init__(self, n_classes, input_dim: int, parameters: dict) -> None:
        self.reservoir_size = parameters["reservoir_size"] 
        self.backwards = parameters["backwards"]

        self.input_weights = parameters["input_scaling"]*(-1 + 2*np.random.rand(parameters["reservoir_size"], input_dim))
        self.reservoir_bias = parameters["bias_scaling"]*(1 + 2*np.random.rand(parameters["reservoir_size"], 1))
        
        self.reservoir_weights = sparse.random(parameters["reservoir_size"], parameters["reservoir_size"], density=parameters["density"], data_rvs = lambda shape: -1 + 2*np.random.rand(shape))
        self.reservoir_weights =  parameters["reservoir_scaling"]*(self.reservoir_weights/(np.max(np.real(np.linalg.eigvals(self.reservoir_weights.toarray())))))
        
        self.readout = MultiReadoutLayer(n_classes, parameters["reservoir_size"]) 

    def predict(self, x):

        reservoir_state = []
        for x_t in x:
            x_t = x_t[:,:,0]
            if len(reservoir_state)==0:
                new_state = np.tanh(self.input_weights @ x_t + self.reservoir_bias)

            else:
                new_state = np.tanh(self.input_weights @ x_t + self.reservoir_weights @ reservoir_state[-1][0] + self.reservoir_bias)
            
            new_state = np.expand_dims(new_state, 0)
            reservoir_state.append(new_state)


        return np.concatenate(reservoir_state, axis=0)




#%% Run experiments here
parameters = {"reservoir_size" : 500,
              "density" : 0.01,
              "input_scaling" : 2,
              "reservoir_scaling" : 1,
              "bias_scaling" : 1,
              "backwards" : False}

data = DataLoader(n_training=6400, n_test=800)
res = ReservoirLayer(8, 64, parameters)

layer = MultiReadoutLayer(8, 64)

layer.train(data.train_set, res) 
readout_layers = layer.readout_layers

#%%
layer = MultiReadoutLayer.from_layers(readout_layers, 8,64)
layer.test(data.test_set, res)
