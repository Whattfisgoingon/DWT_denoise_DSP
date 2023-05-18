import wave
import matplotlib.pyplot as plt
import numpy as np
import pywt
import struct
def denoise_frame(frame, threshold):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(frame, 'haar', level=5)


    # Apply thresholding to the detail coefficients
    denoised_coeffs = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]

    # Reconstruct the denoised frame
    denoised_frame = pywt.waverec([coeffs[0]] + denoised_coeffs, 'haar')

    return denoised_frame

wav_obj =wave.open('E:\learning\DSP_project_code\sound\Recording_short.wav','rb')

sample_freq = wav_obj.getframerate()
n_samples = 10000
t_audio = n_samples/sample_freq
n_channels = wav_obj.getnchannels()

signal_wave = wav_obj.readframes(n_samples)

signal_array = np.frombuffer(signal_wave, dtype=np.int16)

l_channel = signal_array[0::2]
r_channel = signal_array[1::2]

l_channel_denoise = denoise_frame (l_channel, 1000000)

times = np.linspace(0, n_samples/sample_freq, num=n_samples)

plt.subplot(1, 2, 1)
plt.plot(times, l_channel)
plt.title('Left Channel')
plt.ylabel('Signal Value')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)


plt.subplot(1, 2, 2)
plt.plot(times, l_channel_denoise)
plt.title('Left Channel')
plt.ylabel('Signal Value')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)
plt.show()

plt.subplot(1, 2, 1)
plt.specgram(l_channel, Fs=sample_freq, vmin=-20, vmax=50)
plt.title('Left Channel')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)


plt.subplot(1, 2, 2)
plt.specgram(l_channel_denoise, Fs=sample_freq, vmin=-20, vmax=50)
plt.title('Left Channel')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)
plt.show()