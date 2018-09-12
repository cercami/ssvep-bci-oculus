from sklearn.cross_decomposition import CCA
import numpy as np 



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
    samplingRate=178
    
    		
    numpyBuffer = np.array(data)
    size = np.shape(data)
    
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
    
    result = [abs(result1), abs(result2),abs(result3),abs(result4)]        
    ab = max(result,key=float)
                   
    if (abs(result1) == ab):
        value = 5
                    
    if (abs(result2) == ab):
        value=7
                    
    if (abs(result3) == ab):
        value= 9
                    
    if (abs(result4) == ab):
        value = 11
                    
    return value, result
                                
                               	


              
                                
