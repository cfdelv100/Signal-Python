# frequency = number of times a sine wave repeats in one second.
# real world signals are analog, while computers are digital.
# sine wave formula y(t) = A * sin(2 * pi * f * t)
import numpy as np
import wave
import struct
import matplotlib as plt
import matplotlib.pyplot as plt


# f = # of times a wave repeats in 1 second
frequency = 1000
noisy_freq = 50
num_samples = 48000

# the sampling rate of analog to digital conversion

sampling_rate = 48000.0
amplitude = 16000

file = "test.wav"

# setting up sine wave in a list
sine_wave = [np.sin(2 * np.pi * frequency * x/sampling_rate) for x in range(num_samples)]
sine_wave = np.array(sine_wave)

# setting up sine wave noise
sine_noise = [np.sin(2 * np.pi * noisy_freq * x1/sampling_rate) for x1 in range(num_samples)]
sine_noise = np.array(sine_noise)

# combining the signals
combined_signals = sine_wave + sine_noise


nframes = num_samples
comptype = "NONE"
compname = "not compressed"
nchannels = 1
sampwidth = 2

wav_file = wave.open(file, 'w')
wav_file.setparams((nchannels, sampwidth, int(sampling_rate), nframes, comptype, compname))

# s is the single sample of sine_wave, h is a 16-bit number
for s in sine_wave:
    wav_file.writeframes(struct.pack('h', int(s * amplitude)))

# reading the generated file in the previous above
frame_rate = 48000.0
infile = "test.wav"
num_samples = 48000
wav_file = wave.open(infile, 'r')
data = wav_file.readframes(num_samples)
wav_file.close()

# to make this in an unpack num_samples 16 bit words
data = struct.unpack('{n}h'.format(n=num_samples), data)


# taking the fft of the daya and create an array with all frequencies present in the signal
data = np.array(data)
data_fft = np.fft.fft(data)

# generate real numbers of the data
frequencies = np.abs(data_fft)

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

# fft of the combined noise and signal wave
data_fft = np.fft.fft(combined_signals)
freq = (np.abs(data_fft[:len(data_fft)]))

plt.plot(freq)
plt.title("Before Filtering: Will have main signal (1000 Hz) + noise frequency (50 Hz)")
plt.xlim(0, 1200)


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
plt.title("After Filtering: Main signal only(1000 Hz)")
plt.xlim(0, 1200)
plt.show()
plt.close()

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
