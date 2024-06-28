# frequency = number of times a sine wave repeats in one second.
# real world signals are analog, while computers are digital.
# sine wave formula y(t) = A * sin(2 * pi * f * t)
import dash
import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_auth

app = dash.Dash(__name__, suppress_callback_exceptions=True,
                meta_tags=[{'signal':'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

auth = dash_auth.BasicAuth(
    app,
    {'user1': 'password'}
)

# f = # of times a wave repeats in 1 second
frequency = 1000
noisy_freq = 50
num_samples = 48000

# the sampling rate of analog to digital conversion

sampling_rate = 48000.0
amplitude = 16000
# TODO: Find direction to place this as a web application using either flask or dash.
# TODO: Allow user to upload test file and push through app displayed on localhost.
file = "test.wav"
file2 = "test2.wav"

# setting up sine wave in a list
sine_wave = [np.sin(2 * np.pi * frequency * x/sampling_rate) for x in range(num_samples)]
sine_wave = np.array(sine_wave)

# setting up sine wave noise
sine_noise = [np.sin(2 * np.pi * noisy_freq * x1/sampling_rate) for x1 in range(num_samples)]
sine_noise = np.array(sine_noise)

# setting up a cosine wave in a list
cos_wave = [np.cos(2 * np.pi * frequency * x/sampling_rate) for x in range(num_samples)]
cos_wave = np.array(cos_wave)

# setting up a cosine wave noise
cos_noise = [np.cos(2 * np.pi * frequency * x/sampling_rate) for x in range(num_samples)]
cos_noise = np.array(cos_noise)


# combining the signals
combined_signals = sine_wave + sine_noise
combined_cos_signal = cos_wave + cos_noise


nframes = num_samples
comptype = "NONE"
compname = "not compressed"
nchannels = 1
sampwidth = 2

wav_file = wave.open(file, 'w')
wav_file.setparams((nchannels, sampwidth, int(sampling_rate), nframes, comptype, compname))

wav_file2 = wave.open(file2, 'w')
wav_file2.setparams((nchannels, sampwidth, int(sampling_rate), nframes, comptype, compname))

# s is the single sample of sine_wave, h is a 16-bit number
for s in sine_wave:
    wav_file.writeframes(struct.pack('h', int(s * amplitude)))

# c in the single sample of a cos_wave
for c in cos_wave:
    wav_file2.writeframes(struct.pack('h', int(c * amplitude)))

# reading the generated file in the previous sine above
frame_rate = 48000.0
infile = "test.wav"
num_samples = 48000
wav_file = wave.open(infile, 'r')
data = wav_file.readframes(num_samples)
wav_file.close()

# reading the generated file in the previous cos above
frame_rate2 = 48000.0
infile2 = "test2.wav"
num_samples = 48000
wav_file = wave.open(infile2, 'r')
data2 = wav_file.readframes(num_samples)
wav_file2.close()

# to make this in an unpack num_samples 16 bit words
data = struct.unpack('{n}h'.format(n=num_samples), data)

# unpack data2
data2 = struct.unpack('{n}h'.format(n=num_samples), data2)

# taking the fft of the daya and create an array with all frequencies present in the signal
data = np.array(data)
data_fft = np.fft.fft(data)

# data2
data2 = np.array(data2)
data_fft2 = np.fft.fft(data2)


# generate real numbers of the data
frequencies = np.abs(data_fft)

# data 2 real numbers
frequencies2 = np.abs(data_fft2)

# print the array element with the highest value
print("The frequency is {} Hz.".format(np.argmax(frequencies)))

# set up plots for the wave
plt.subplot(4, 1, 1)
plt.plot(data[:300])
plt.title("Original audio wave")
plt.subplots_adjust(hspace=5)
plt.plot(sine_wave[:500])

plt.subplot(4, 1, 2)
plt.title("Noisy wave")
plt.plot(sine_noise[:4000])

plt.subplot(4, 1, 3)
plt.title("Original + Noise")
plt.plot(combined_signals[:3000])

plt.subplot(4, 1, 4)
plt.plot(frequencies)
plt.title("Frequencies found")
plt.xlim(0, 1200)
plt.show()
plt.close()

# fft of the combined noise and signal wave
data_fft = np.fft.fft(combined_signals)
freq = (np.abs(data_fft[:len(data_fft)]))

plt.plot(freq)
plt.title("Before Filtering: Will have main signal (1000 Hz) + noise frequency (50 Hz)")
plt.xlim(0, 1200)
plt.close()

# cosine graph box
print("The frequency is {} Hz.".format(np.argmax(frequencies2)))
plt.subplot(4, 1, 1)
plt.plot(data2[:300])
plt.title("Original Cosine Wave")
plt.subplots_adjust(hspace=5)
plt.plot(cos_wave[:500])

plt.subplot(4, 1, 2)
plt.title("Noisy Cosine Wave")
plt.plot(cos_noise[:4000])

plt.subplot(4, 1, 3)
plt.title("Combined Cosine Wave")
plt.plot(combined_cos_signal[:3000])

plt.subplot(4, 1, 4)
plt.plot(frequencies2)
plt.title("Frequencies Found")
plt.xlim(0, 1200)

# combined fft of the combined cosine and wave
data_fft2 = np.fft.fft(combined_cos_signal)
freq2 = (np.abs(data_fft2[:len(data_fft2)]))

plt.plot(freq2)
plt.title("Number of Frequencies")
plt.xlim(0, 1200)

plt.show()
plt.close()



# filtering
filtered_freq = []
index = 0
for f in freq:
    # filter between lower and upper limits, choosing 950 as closest to 1000, not real world like
    if 950 < index < 1050:
        # has a real value, and we will try > 1,
        if f > 1:
            filtered_freq.append(f)
        else:
            filtered_freq.append(0)
    else:
        filtered_freq.append(0)
index += 1

# plotting the filtered frequency
plt.plot(filtered_freq)
plt.title("Before filtering: Will have main signal (1000Hz) + noise frequency (50Hz)")
plt.xlim(0, 1200)
plt.show()


recovered_signal = np.fft.ifft(filtered_freq)
plt.subplot(3, 1, 1)
plt.title("Original Sine Wave")
plt.subplots_adjust(hspace=5)
plt.plot(sine_wave[:500])

plt.subplot(3, 1, 2)
plt.title("Noisy wave")
plt.plot(combined_signals[:4000])

plt.subplot(3, 1, 3)
plt.title("Sine wave after clean up")
plt.plot((recovered_signal[:500]))
plt.show()

plt.close()
