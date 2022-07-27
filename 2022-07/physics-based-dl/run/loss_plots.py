"""
Created on Wed Jul 27 21:38:58 2022

@author: ABHIJEET
"""

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams



rcParams["figure.dpi"] = 180
rcParams["font.size"] = 10

dnnFoam_file = "./dnnPotentialFoam-00000021.csv"      # DNN run with optimum hyperparameters
pinnPotentialFoam_file = "./pinnPotentialFoam-00000009.csv"   # PiNN run with optimum hyperparameters

#DNN plots
data_frame_dnn = pd.read_csv(dnnFoam_file)
plt.plot(data_frame_dnn["EPOCH"], data_frame_dnn["TRAINING_MSE"],'r', label="TRAINING_MSE")
plt.loglog()
plt.legend()
plt.title("Evolution of losses: DNN")
plt.xlabel("Epochs")
plt.ylabel("Errors")
plt.show()

#PiNN plots 
data_frame_potential = pd.read_csv(pinnPotentialFoam_file)
plt.plot(data_frame_potential["EPOCH"], data_frame_potential["DATA_MSE"],'r', label="DATA_MSE")
plt.plot(data_frame_potential["EPOCH"], data_frame_potential["GRAD_MSE"],'k-.', label="GRAD_MSE")
plt.plot(data_frame_potential["EPOCH"], data_frame_potential["TRAINING_MSE"],'m', label="TRAINING_MSE")
plt.loglog()
plt.legend()
plt.title("Evolution of losses: PINN")
plt.xlabel("Epochs")
plt.ylabel("Errors")
plt.show()


#comparison between Data_MSE_DNN and Data_MSE_pinn
plt.plot(data_frame_dnn["EPOCH"], data_frame_dnn["DATA_MSE"], 'r',label="DATA_MSE_DNN")
plt.plot(data_frame_potential["EPOCH"], data_frame_potential["DATA_MSE"],'k-.', label="DATA_MSE_PINN")
plt.loglog()
plt.legend()
plt.title("Comparison between MSE loss in DNN and PINN")
plt.xlabel("Epochs")
plt.ylabel("Errors")
plt.show()