import numpy as np
import pyaudio
import struct
from scipy.fftpack import fft
import time
import torch
import torch.nn as nn

class AudioStream(object):
    def __init__(self):

        # stream constants
        self.CHUNK = 1024 
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False
        
        #Abtastrate in Sekunden pro Abfrage
        self.DELAYS = 0.01
        #Frequenzen, die geprüft werden sollen
        self.FREQUENCIES = np.array([
            200., 500., 700.])
        self.FREQUENCIES *= float(self.CHUNK/self.RATE)
        #Später Zwiscchenspeicherung der einzelnen Lautstärken
        self.values = [np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3),
                       np.zeros(3)]

        # stream object
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            input_device_index=2,
            output_device_index=4,
            frames_per_buffer=self.CHUNK,
            
        )
        self.start()

    def start(self):

        print('stream started')
        frame_count = 0
        start_time = time.time()
        model = load_model()

        while not self.pause:
            s_time = time.perf_counter()

            data = self.stream.read(self.CHUNK)
            data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)

            # compute FFT and update line
            yf = np.fft.fft(data_int) / (128 * self.CHUNK)
            
            self.FREQUENCIES *= float(self.CHUNK / self.RATE)
            export = np.abs(yf[0:self.CHUNK]) 
            print(export[int(260 * self.CHUNK / self.RATE)])
           

            freq = np.array([float(export[int(i)]) for i in self.FREQUENCIES])
            freq *= 10
            self.values.pop(0)
            self.values.append(freq)
            prediction = predict(torch.from_numpy(np.array(self.values).transpose().astype(np.float32)), model)
            #print(prediction)

            if np.any(prediction != 0):
                print("Arc")
            
            frame_count += 1

            if time.time() - start_time >= 200:
                self.pause = True

            if time.perf_counter()-s_time <= self.DELAYS:
                time.sleep(self.DELAYS-(time.perf_counter()-s_time))
            self.stream.write(data)
        else:
            self.fr = frame_count / (time.time() - start_time)
            print('average operation rate = {:.0f} per second'.format(self.fr))
            self.exit_app()

    def exit_app(self):
        print('stream closed')
        self.p.close(self.stream)

def predict(input, model):
    #inputs = inputs.to(self.device) # You can move your input to gpu, torch defaults to cpu

    # Run forward pass
    with torch.no_grad():
        pred = model(input)

    # Do something with pred
    #pred = pred.detach().cpu().numpy()
    x = 1/(1+np.exp(-pred.numpy()))
    return x.astype(int)

def load_model() -> object:
    output_size = 1
    input_size = 13

    class MLP(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.layers = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)  # Ausgabe für binäre Klassifikation
                )
            def forward(self, x):
                #x = x - feature_means
                out = self.layers(x)
                return out
            
    model = MLP(input_size)
    model.load_state_dict(torch.load('model_cool.pt', weights_only=True))
    return model

if __name__ == '__main__':
    AudioStream()