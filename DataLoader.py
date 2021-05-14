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
        commands = commands[commands != 'README.md']
        print('Commands:', commands)

        filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        print('Number of total examples:', num_samples)
        print('Example file tensor:', filenames[0])

        train_files = filenames[:500]
        val_files = filenames[6400: 6400 + 800]
        test_files = filenames[-800:]

        print('Training set size', len(train_files))
        print('Validation set size', len(val_files))
        print('Test set size', len(test_files))     

        AUTOTUNE = tf.data.AUTOTUNE
        files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.train_set = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

        files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.val_set = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

        files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.test_set = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

#Utility methods
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)
 
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] 

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform):
    equal_length = tf.concat([waveform, zero_padding], 0)

    spectrogram = tf.signal.stft(equal_length, frame_length=512, frame_step=264, fft_length=32)
    spectrogram = tf.math.abs(spectrogram)
    
    spectrogram = tf.math.pow(spectrogram, 0.2)
    real = tf.math.sqrt(tf.math.abs(tf.math.real(spectrogram)))
    imag = tf.math.sqrt(tf.math.abs(tf.math.imag(spectrogram))) 
    spectrogram = tf.math.abs(tf.math.sin(real)) - tf.math.abs(tf.math.cos(imag))
    spectrogram = spectrogram / tf.reduce_max(spectrogram) 
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id