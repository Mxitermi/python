import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import TclError

# Constants
CHUNK = 1024 * 8             # Samples per frame
FORMAT = pyaudio.paInt16     # Audio format
CHANNELS = 2                 # Single channel for microphone
RATE = 44100                 # Samples per second

# PyAudio class instance
p = pyaudio.PyAudio()
# Stream object to get data from microphone
inputstream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,output=True,frames_per_buffer=CHUNK)
# Stream object to output antisound
outputstream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,output=True,frames_per_buffer=CHUNK)

frame_count = 0
start_time = time.time()
plt.show()
while True:

    # Binary input data
    data = inputstream.read(CHUNK)
    # Convert data to integers
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
    # Convert data to np array 
    data_np = np.array(data_int, dtype='b')[::2]
    
    # FFT the data
    fft_data = np.fft.fft(data_np)
    freqs = np.fft.fftfreq(len(data_np),d = 1./RATE) #takes the frequencies of the data and stores them in an array
    mag_fft_data = np.abs(fft_data) #fourier coefficients are complex numbers that tell you the magnitude of the wave. Use np.abs to find magnitude of complex number and then store those in the array
    threshold = freqs(np.argmax(mag_fft_data)) #finds frequency of max amplitude 
    indices_to_zero = np.where((np.abs(freqs) < np.abs(threshold) - 40) | (np.abs(freqs) > np.abs(threshold) + 40)) #bandpass filter
    '''
    threshold  = max(mag_fft_data)/2 #finding the maximum amplitude so far, and creating a threshold that is half of the max
    indices_to_zero = np.where(mag_fft_data < threshold) #finds every point in the array where the data is less than the threshold
    '''
    fft_data_clean = np.copy(fft_data) #copies th fourier data for clean up
    fft_data_clean[indices_to_zero] = 0 #makes all the points where the magnitude was below the threshold 0
    np_int_band_pass = np.fft.ifft(fft_data_clean) #inverse transforms it to put back into function
    list_int_band_pass = np_int_band_pass.tolist() #converts np array to python list
    bin_band_pass = struct.pack(str(CHUNK) + 'h', *list_int_band_pass) #converts to binary
    # Print
    print('Frequencies: ', freqs)
    print('Fourier Coefficients: ', fft_data)

    outputstream.write(bin_band_pass)