# -*- coding: utf-8 -*-
"""
Template for the DemoScene
"""

import mules            # The signal acquisition toolbox we'll use (MuLES)
import numpy as np      # Module that simplifies computations on matrices 
import bci_workshop_tools as BCIw # Bunch of useful functions for the workshop
import experiment
from python_VR_version import CalculaCCA

# MuLES connection parameters    
mules_ip = '127.0.0.1'
mules_port = 30000

# Creates a mules_client
mules_client = mules.MulesClient(mules_ip, mules_port) 
params = mules_client.getparams() # Get the device parameters   
fs = params['sampling frequency']

#%% Set the experiment parameters
eeg_buffer_secs = 15  # Size of the EEG data buffer used for plotting the 
                      # signal (in seconds) 
win_test_secs = 3     # Length of the window used for computing the features 
                      # (in seconds)
shift_secs = 0.5      # Shift between two consecutive windows (in seconds)
indexes_channel = [1,2,3,4]     # Index of the channnel to be used 
                                 
#%% Initialize the buffers for storing raw EEG and features
# Initialize raw EEG data buffer (for plotting)
eeg_buffer = np.zeros((fs*eeg_buffer_secs, len(indexes_channel))) 
experiment.tone(500,500)
                                            
#%% Start pulling data    
mules_client.flushdata()  # Flush old data from MuLES       
  
# The try/except structure allows to quit the while loop by aborting the 
# script with <Ctrl-C>
print(' Press Ctrl-C in the console to break the While Loop')
try:    
    
    # The following loop does what we see in the diagram of Exercise 1:
    # acquire data, compute features, visualize the raw EEG and the features        
    while True:    
        
        """ 1- ACQUIRE DATA """
        eeg_data = mules_client.getdata(shift_secs, False)   # Obtain EEG data from MuLES
        eeg_data = eeg_data[:, indexes_channel]              # Removes unwanted channels            
        eeg_buffer = BCIw.update_buffer(eeg_buffer, eeg_data)[0] # Update EEG buffer
        
        # Select the last win_test_secs seconds to perform task
        test_eeg = eeg_buffer[(-win_test_secs*fs): , :]
        
        """ 2- COMPUTE FEATURES """
        ft = np.mean(test_eeg)
        (value, result) = CalculaCCA(test_eeg.T)
        
       
        """ 3- VISUALIZE THE RAW EEG AND THE FEATURES """       
        print(str(value) + ' Hz')
        print(result)
        
                 
except KeyboardInterrupt:    
    mules_client.disconnect() # Close connection

 
