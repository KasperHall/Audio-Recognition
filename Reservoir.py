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


# %% Utility methods


# %%


# %% Correct ridge_regression


print(spectrogram_ds.element_spec)

n_samples= 6400
design_matrix = np.zeros((n_samples, 59*17))
target_output = np.zeros((n_samples, 8))
for i, (spectrogram, label_id) in enumerate(spectrogram_ds):
    spectrogram = np.array(spectrogram)
    spectrogram = spectrogram.flatten()
    design_matrix[i, :] = spectrogram

    target_output[i, label_id] = 1

output_weights = np.linalg.inv(design_matrix.T @ design_matrix + np.eye(59*17)) @ design_matrix.T @ target_output[:,:]
print(output_weights.shape)

# %%
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(test_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

correct = 0
for i, (spectrogram, label_id) in enumerate(spectrogram_ds):
    
    spectrogram = np.array(spectrogram)
    spectrogram = spectrogram.flatten()
    result = np.argmax(spectrogram @ output_weights)
    
    if label_id==result:
        correct += 1

print(correct/len(spectrogram_ds))


# %% build training tensors
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# T: batch*250*10 matrix
# V: batch*250*64
T = []
V = []
for spectrogram, label_id in spectrogram_ds:
    t = np.zeros((247, 8, 1), dtype=np.float32)
    t[:, label_id, :] = 1
    T.append(t) 
    V.append(spectrogram)

T = tf.concat(T, axis=-1)
V = tf.concat(V, axis=-1)

print(tf.shape(V))
W = T @ tf.linalg.pinv(V)

# %% Test
files_ds = tf.data.Dataset.from_tensor_slices(test_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# T: batch*250*10 matrix
# V: batch*250*64
T = []
V = []

correct = 0
for spectrogram, label_id in spectrogram_ds:
    spectrogram = tf.transpose(spectrogram())
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

reservoir_size = 600
reservoir_density = 0.2
reservoir_weights = sparse.random(reservoir_size, reservoir_size, reservoir_density, data_rvs=lambda shape : -1 + 2*np.random.rand(shape))
eigenvals = np.linalg.eigvals(reservoir_weights.toarray())
reservoir_weights = 2*reservoir_weights/(np.real(max(eigenvals)))
print(max(np.linalg.eigvals(reservoir_weights.toarray())))

input_weights = -0.1 + 0.2*np.random.rand(reservoir_size, 129)


for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(500)):
    t = np.zeros((8,1), dtype=np.float64)
    t[label_id, :] = 1
    T.append(t) 
    
    reservoir_state = np.zeros([reservoir_size, 1])
    for x in spectrogram:
        reservoir_state = np.tanh(input_weights.dot(x) + reservoir_weights.dot(reservoir_state))
    V.append(reservoir_state)
    if i%500==0:
        print(i)

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
for spectrogram, label_id in spectrogram_ds:

    reservoir_state = np.zeros([reservoir_size, 1])
    for x in spectrogram:
        reservoir_state = np.tanh(input_weights.dot(x) + reservoir_weights.dot(reservoir_state))

    res = W @ reservoir_state
    res = tf.argmax(res)
    if (label_id == res):
        correct += 1

correct/len(spectrogram_ds)

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
reservoir_weights = 2*reservoir_weights/(np.real(max(eigenvals)))
print(max(np.linalg.eigvals(reservoir_weights.toarray())))
reservoir_weights=reservoir_weights.toarray()

input_weights = -0.1 + 0.2*np.random.rand(reservoir_size, 129)


for j, (spectrogram, label_id) in enumerate(spectrogram_ds.take(3000)):
    t = np.zeros((247,8,1), dtype=np.float64)
    t[:,label_id, :] = 1
    T.append(t) 
    
    reservoir_state = np.zeros([reservoir_size, 248, 1])
    for i, x in enumerate(spectrogram):
        reservoir_state[:,i+1,:] = np.tanh(input_weights.dot(x) + reservoir_weights.dot(reservoir_state[:,i]))
    if j%100==0:
        print(j)
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
for spectrogram, label_id in spectrogram_ds:

    reservoir_state = np.zeros([reservoir_size, 248, 1])
    for i, x in enumerate(spectrogram):
        reservoir_state[:,i+1,:] = np.tanh(input_weights.dot(x) + reservoir_weights.dot(reservoir_state[:,i]))
    
    reservoir_state = tf.transpose(reservoir_state, (1,0,2))
    res = W @ reservoir_state[1:,:,:]
    res = tf.argmax(tf.reduce_mean(res, axis=0))
    if (label_id == res):
        correct += 1

correct/len(spectrogram_ds)
