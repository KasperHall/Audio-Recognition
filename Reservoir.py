#%% Imports
from ReadoutLayer import SingleReadoutLayer
from DataLoader import DataLoader


#%%Load data
data = DataLoader()

#%%Construct readoutLayers

l = SingleReadoutLayer(8, 59*17)
l.train(data.train_set, 59, 17)

#%% 
print(l.test(data.train_set))
print(l.test(data.test_set))

