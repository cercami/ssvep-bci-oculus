from sklearn.cross_decomposition import CCA
import numpy as np 
from scipy.signal import butter, lfilter



def butter_bandstop_filter(data, lowcut, highcut, fs, order):


    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def getReferenceSignals(length, target_freq, samplingRate):
    reference_signals = []
    t = np.arange(0, (length/samplingRate), step=1.0/samplingRate)
    reference_signals.append(np.sin(np.pi*2*target_freq*t))
    reference_signals.append(np.cos(np.pi*2*target_freq*t))
    reference_signals.append(np.sin(np.pi*4*target_freq*t))
    reference_signals.append(np.cos(np.pi*4*target_freq*t))
    reference_signals = np.array(reference_signals)
    return reference_signals

                                
def CalculaCCA(data):
    samplingRate=500
    
    
    data_filtered = butter_bandpass_filter(data,4.0,35.0,samplingRate)
    data_filtered = data 
    
    data_notch = butter_bandstop_filter(data_filtered,58.0, 62.0,samplingRate,4)
    		
    numpyBuffer = np.array(data_notch)
    size = np.shape(data_notch)
    
    freq1= getReferenceSignals(size[1],5,samplingRate)
    freq2= getReferenceSignals(size[1],7,samplingRate)
    freq3= getReferenceSignals(size[1],9,samplingRate)
    freq4= getReferenceSignals(size[1],11,samplingRate)
    
    cca=CCA(n_components=1)
    			 	
    cca.fit(numpyBuffer.T,freq1.T)
    O1_a,O1_b = cca.transform(numpyBuffer.T, freq1.T)
    result1 = np.corrcoef(O1_a.T,O1_b.T)[0,1]
                    
                    
    cca.fit(numpyBuffer.T,freq2.T)
    O1_a,O1_b = cca.transform(numpyBuffer.T, freq2.T)
    result2 = np.corrcoef(O1_a.T,O1_b.T)[0,1]
    
                    
    cca.fit(numpyBuffer.T,freq3.T)
    O1_a,O1_b = cca.transform(numpyBuffer.T, freq3.T)
    result3 = np.corrcoef(O1_a.T,O1_b.T)[0,1]
                    
    cca.fit(numpyBuffer.T,freq4.T)
    O1_a,O1_b = cca.transform(numpyBuffer.T, freq4.T)
    result4 = np.corrcoef(O1_a.T,O1_b.T)[0,1]
    
    result = [abs(result1),abs(result2), abs(result3), abs(result4)]        
    ab = max(result,key=float)
                   
    if (abs(result1) == ab):
        value = 5
                    
    if (abs(result2) == ab):
        value= 7
                    
    if (abs(result3) == ab):
         value= 9
                    
    if (abs(result4) == ab):
        value = 11
                    
    return value
                                
                               	


              
                                
