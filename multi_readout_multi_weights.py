# %% Imports
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

    resize = preprocessing.Resizing(32, 32)
    norm = preprocessing.Normalization()
    def __init__(self) -> None:
    
        data_dir = pathlib.Path('data/mini_speech_commands')
        if not data_dir.exists():
            tf.keras.utils.get_file(
                'mini_speech_commands.zip',
                origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract=True,
                cache_dir='.', cache_subdir='data')

        commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        self.commands = commands[commands != 'README.md']
        print('Commands:', commands)

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
        self.train_set = waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

        files_ds = tf.data.Dataset.from_tensor_slices(val_files)
        waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.val_set = waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

        files_ds = tf.data.Dataset.from_tensor_slices(test_files)
        waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.test_set = waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


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
    def __init__(self, n_classes: int, n_features: int, ridge_parameter = 0.1) -> None:
        self.output_weights = np.zeros((n_features, n_classes))
        self.n_classes = n_classes
        self.n_features = n_features
        self.ridge_parameter = ridge_parameter



    def predict(self, x):
        x = x.numpy().flatten()
        return np.argmax(x @ self.output_weights)

    def test(self, test_set):
        correct = 0
        for x, target in test_set:
            if self.predict(x) == target:
                correct += 1
                
        return correct/len(test_set)

    def train(self, training_set, n_steps, n_freq):

        n_samples = len(training_set)
        
        design_matrix = np.zeros((n_samples, n_steps*n_freq))
        target_output = np.zeros((n_samples, self.n_classes))

        for i, (x, target) in enumerate(training_set):
            design_matrix[i, :] = x.numpy().flatten()
            target_output[i, target] = 1

        self.output_weights[:, :] = np.linalg.inv(design_matrix.T @ design_matrix + self.ridge_parameter*np.eye(n_steps*n_freq)) @ design_matrix.T @ target_output

#%% MultiReadoutLayer
class MultiReadoutLayer():
    """
    Predicts the most likely output class from a series of vectors through ridge regression
    """

    def __init__(self, n_classes: int, n_steps: int, n_features: int, ridge_parameter = 0.1) -> None:
        self.readout_layers = [SingleReadoutLayer(n_classes, n_features, ridge_parameter) for i in range(n_steps)]
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_steps = n_steps
        self.ridge_parameter = ridge_parameter



    def predict(self, x):
        result = np.array([x_t.numpy().T @ l.output_weights for l, x_t in zip(self.readout_layers, x)])
        return np.argmax(np.mean(result, 0))
    
    def test(self, test_set):
        correct = 0
        for x, target in test_set:
            if self.predict(x) == target:
                correct += 1
                
        return correct/len(test_set)
    
    def train(self, training_set, n_steps, n_freq):
        for t, layer in enumerate(self.readout_layers):
            sub_training_set = [(x[t], label_id) for x, label_id in training_set]
            layer.train(sub_training_set, 1, self.n_features)
            if t%10==1:
                print(f"{t}/{self.n_steps} trained.")

#%% Run experiments here
data = DataLoader()
#data.visualize()

#%%
layer = MultiReadoutLayer(8, 32, 32)
layer.train(data.train_set, 32, 32) 
layer.test(data.test_set)

#%% Test training samples impact

results = []
for ridge_parameter in [0, 0.1, 0.5, 1, 2]:
    layer = MultiReadoutLayer(8, , 129, ridge_parameter)
    layer.train(data.train_set.take(1000), 124, 129) 
    results.append(layer.test(data.test_set))

#%%
print(results)
plt.plot([0, 0.1, 0.5, 1, 2], results)
plt.show()
