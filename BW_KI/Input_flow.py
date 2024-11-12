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
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False
        
        #Abtastrate in Sekunden pro Abfrage
        self.DELAYS = 0.05
        #Frequenzen, die geprüft werden sollen
        self.FREQUENCIES = np.array([
            400.,
            440.,
            200.,
            300.,
            400.,
            500.,
            700.,
            1000.,
            1500.,
            2000.,
            3000.,
            4000.,
            6000.])
        self.FREQUENCIES *= float(self.CHUNK/self.RATE)
        #Später Zwiscchenspeicherung der einzelnen Lautstärken
        self.values = [np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13),
                       np.zeros(13)]

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
            yf = fft(data_int)
            export = np.abs(yf[0:self.CHUNK]) / (128 * self.CHUNK)
            
            freq = np.array([float(export[int(i)]) for i in self.FREQUENCIES])
            self.values.pop(0)
            self.values.append(freq)
            prediction = predict(torch.from_numpy(np.array(self.values).transpose().astype(np.float32)), model)

            if 1 in prediction:
                print("Rck")

            frame_count += 1

            if time.time() - start_time >= 200:
                self.pause = True

            if time.perf_counter()-s_time <= self.DELAYS:
                time.sleep(self.DELAYS-(time.perf_counter()-s_time))

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
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_size)
                )
            def forward(self, x):
                #x = x - feature_means
                out = self.layers(x)
                return out
            
    model = MLP(input_size)
    model.load_state_dict(torch.load('model_real.pt', weights_only=True))
    return model

if __name__ == '__main__':
    AudioStream()