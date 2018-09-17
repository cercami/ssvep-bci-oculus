# -*- coding: utf-8 -*-
"""
Functions for experiments
"""

import winsound
import time    
import socket
import array
import struct
import numpy as np

def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n

def psd_fft(x, fs):
    '''
    x has to have the shape [samples, channels]
    '''
    # 1. Compute the PSD
    winSampleLength, nbCh = x.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    x = x - np.mean(x, axis=0)  # Remove offset
    x = (x.T*w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(x, n=NFFT, axis=0)/winSampleLength
    PSD = 2*np.abs(Y[0:int(NFFT/2), :])
    f_ax = fs/2*np.linspace(0, 1, int(NFFT/2))
    return PSD, f_ax

def tone(f=500, d=500):
    """
    Uses the Sound-playing interface for Windows to play a beep
        
    Arguments
    f: Frequency of the beep in Hz
    d: Duration of the beep in ms
    """
    winsound.Beep(f,d)
    
def pause(seconds):
    """
    Pauses the execution for s seconds
    
    Arguments
    s: Number of seconds to wait
    """
    time.sleep(seconds)

class TcpClient():
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
    
    def connect(self):
        """
        Connects to a TCP/IP server 
        """
        print('Attempting connection')
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.ip, self.port))
            print('Connection successful')
        except:
            self.client = None
            print('Connection attempt unsuccessful')
            raise
            
    def writeInt32(self, integer32):
        """
        Writes one Int32
        """
        bytes_4B = struct.pack('i', integer32)
        bytes_4B = bytes_4B[::-1]
        self.client.send(bytes_4B)
        return
		
    def writeArray(self, array):
        bytes = struct.pack('=%sf' % array.shape[0], *array)
        bytes = bytes[::-1] # reverse order
        self.client.send(bytes)
        return
    
    def readInt32(self):
        """
        Reads one Int32
        """
        n_bytes_4B = array.array('B',self.client.recv(4) )
        integer32 = struct.unpack('i',n_bytes_4B[::-1])[0] 
        return integer32
    
    def close(self):
        """
        Closes the communication with the Server
        """
        self.client.close()
        self.client = None
        return
