#import tensorflow as tf
#from tensorflow import keras
import predictionNet as prediction
import csv
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
#model = keras.models.load_model("./libs/predModel/saved_models/TFTModel.hf5")
model = prediction.predictNet()
data = pd.read_csv("./HI_lin_degrade.csv")

print(data)
HI = savgol_filter(data['health_index'][:99], 53, 3)
data['health_index'][:99] = HI
print(data)
#prediction = model.predict(data)
#print("p10: ", p10, " p50: ", p50, " p90: ", p90)
#print(prediction)
p10, p50, p90 = model.predict(data)
p10.to_csv("p10test.csv")
p50.to_csv("p50test.csv")
p90.to_csv("p90test.csv")
print("p10: ", p10, " p50: ", p50, " p90: ", p90)
#print(prediction)
