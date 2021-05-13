# %% imports
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

# %% load data
data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

# %% define data sets
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
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

# %% Utility methods

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this 
    # to work in a TensorFlow graph.
    return parts[-2] 

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform):
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)

  spectrogram = tf.signal.stft(equal_length, frame_length=250, frame_step=64)
  spectrogram = tf.math.abs(spectrogram)
  
  spectrogram = tf.math.pow(spectrogram, 0.2)
#   real = tf.math.sqrt(tf.math.abs(tf.math.real(spectrogram)))
#   imag = tf.math.sqrt(tf.math.abs(tf.math.imag(spectrogram))) 
#   spectrogram = tf.math.abs(tf.math.sin(real)) - tf.math.abs(tf.math.cos(imag))
  #spectrogram = spectrogram / tf.reduce_max(spectrogram) 

  return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


# %% build training tensors

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# T: batch*250*10 matrix
# V: batch*250*64
T = []
V = []
for spectrogram, label_id in spectrogram_ds.take(10):
    t = np.zeros((247, 8, 1), dtype=np.float32)
    t[:, label_id, :] = 1
    T.append(t) 
    V.append(spectrogram)

T = tf.concat(T, axis=-1)
V = tf.concat(V, axis=-1)

print(tf.shape(V))
W = T @ tf.linalg.pinv(V)
# %% Test

files_ds = tf.data.Dataset.from_tensor_slices(val_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# T: batch*250*10 matrix
# V: batch*250*64
T = []
V = []

correct = 0
for spectrogram, label_id in spectrogram_ds:
    res = W @ spectrogram
    res = tf.argmax(tf.math.reduce_mean(res, axis=0))
    if (label_id == res):
        correct += 1

correct/len(spectrogram_ds)


# %% Reservoir
import scipy.sparse as sparse

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# T: batch*250*10 matrix
# V: batch*250*64
T = []
V = []

reservoir_size = 512
reservoir_density = 0.2
reservoir_weights = sparse.random(reservoir_size, reservoir_size, reservoir_density, data_rvs=lambda shape : -1 + 2*np.random.rand(shape))
eigenvals = np.linalg.eigvals(reservoir_weights.toarray())
reservoir_weights = reservoir_weights/(np.real(max(eigenvals))*1.01)
print(max(np.linalg.eigvals(reservoir_weights.toarray())))

input_weights = -1 + 2*np.random.rand(reservoir_size, 129)


for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(2000)):
    t = np.zeros((8,1), dtype=np.float64)
    t[label_id, :] = 1
    T.append(t) 
    
    reservoir_state = np.zeros([reservoir_size, 1])
    for x in spectrogram:
        reservoir_state = np.tanh(input_weights.dot(x) + reservoir_weights.dot(reservoir_state))
    V.append(reservoir_state)
    
T = tf.concat(T, axis=-1)
V = tf.concat(V, axis=-1)


W = T @ tf.linalg.pinv(V)

# %% Test reservoir
files_ds = tf.data.Dataset.from_tensor_slices(test_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

T = []
V = []

correct = 0
for spectrogram, label_id in spectrogram_ds.take(500):

    reservoir_state = np.zeros([reservoir_size, 1])
    for x in spectrogram:
        reservoir_state = np.tanh(input_weights.dot(x) + reservoir_weights.dot(reservoir_state))

    res = W @ reservoir_state
    res = tf.argmax(res)
    if (label_id == res):
        correct += 1

correct/500
#len(spectrogram_ds)

# %% Reservoir save all states
import scipy.sparse as sparse

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# T: batch*250*10 matrix
# V: batch*250*64
T = []
V = []

reservoir_size = 512
reservoir_density = 0.2
reservoir_weights = sparse.random(reservoir_size, reservoir_size, reservoir_density, data_rvs=lambda shape : -1 + 2*np.random.rand(shape))
eigenvals = np.linalg.eigvals(reservoir_weights.toarray())
reservoir_weights = reservoir_weights/(np.real(max(eigenvals))*1.01)
print(max(np.linalg.eigvals(reservoir_weights.toarray())))
reservoir_weights=reservoir_weights.toarray()

input_weights = -1 + 2*np.random.rand(reservoir_size, 129)


for j, (spectrogram, label_id) in enumerate(spectrogram_ds.take(10)):
    t = np.zeros((247,8,1), dtype=np.float64)
    t[:,label_id, :] = 1
    T.append(t) 
    
    reservoir_state = np.zeros([reservoir_size, 248, 1])
    for i, x in enumerate(spectrogram):
        reservoir_state[:,i+1,:] = np.tanh(input_weights.dot(x) + reservoir_weights.dot(reservoir_state[:,i]))
    
    V.append(reservoir_state[:,1:,:])

    
V = tf.concat(V, axis=-1)
V = tf.transpose(V, (1,0,2))

T = tf.concat(T, axis=-1)


W = T @ tf.linalg.pinv(V)

# %% Test save all reservoir
files_ds = tf.data.Dataset.from_tensor_slices(test_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

T = []
V = []

correct = 0
for spectrogram, label_id in spectrogram_ds.take(500):

    reservoir_state = np.zeros([reservoir_size, 1])
    for x in spectrogram:
        reservoir_state = np.tanh(input_weights.dot(x) + reservoir_weights.dot(reservoir_state))

    res = W @ reservoir_state
    res = tf.argmax(res)
    if (label_id == res):
        correct += 1

correct/500
#len(spectrogram_ds)