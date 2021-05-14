import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

class DataLoader():
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

    @staticmethod
    def get_spectrogram(waveform):
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)

        spectrogram = tf.signal.stft(equal_length, frame_length=512, frame_step=264, fft_length=32)
        spectrogram = tf.math.abs(spectrogram)
        
        spectrogram = tf.math.pow(spectrogram, 0.2)
        real = tf.math.sqrt(tf.math.abs(tf.math.real(spectrogram)))
        imag = tf.math.sqrt(tf.math.abs(tf.math.imag(spectrogram))) 
        spectrogram = tf.math.abs(tf.math.sin(real)) - tf.math.abs(tf.math.cos(imag))
        spectrogram = spectrogram / tf.reduce_max(spectrogram) 
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


