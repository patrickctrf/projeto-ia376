import os

import numpy as np
import torch
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile


def unscale_data(array, means=0.0, stds=3000.0):
    return array * stds + means


def polar_to_rect(amplitude, angle):
    return amplitude * (np.cos(angle) + 1j * np.sin(angle))


# generator = torch.load("generator.pth").to(torch.device("cpu"))
#
# ruido_e_classes = torch.cat(
#     (
#         torch.normal(mean=0, std=1.0, size=(9, 1)),
#         torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ]).view(-1, 1),
#         torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).view(-1, 1),
#     ), dim=0).view(1, 32, 1)
#
# audio_sintetizado = generator(ruido_e_classes).view(-1)
#
# sd.play(unscale_data(audio_sintetizado.detach().numpy(), means=0.0, stds=300.0), samplerate=16000, blocking=True)


# #===============================================================================
# import pyaudio
# import wave
#
# # define stream chunk
# chunk = 1024
#
# # open a wav format music
# f = wave.open(os.path.join("/home/patrickctrf/Documentos/pode_apagar/projeto-ia376/E2/nsynth-train/audio/guitar_acoustic_000-025-050", ) + ".wav", "rb")
# # instantiate PyAudio
# p = pyaudio.PyAudio()
# # open stream
# stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
#                 channels=f.getnchannels(),
#                 rate=f.getframerate(),
#                 output=True)
# # read data
# data = f.readframes(chunk)
#
# # play stream
# while data:
#     stream.write(data)
#     data = f.readframes(chunk)
#
# # stop stream
# stream.stop_stream()
# stream.close()
#
# # close PyAudio
# p.terminate()
#
# #===============================================================================

sample_audio_array = wavfile.read(os.path.join("/home/patrickctrf/Documentos/pode_apagar/projeto-ia376/E2/nsynth-train/audio/guitar_acoustic_000-025-050", ) + ".wav")[1]
# wavfile.write("/home/patrickctrf/Documentos/pode_apagar/projeto-ia376/E2/nsynth-train/guitar_acoustic_000-025-050.wav", 16000, sample_audio_array)
sample_audio_array = np.pad(sample_audio_array, [(700, 700), ], mode='constant')

sd.play(sample_audio_array, samplerate=16000, blocking=True)

f, t, espectro = signal.stft(x=sample_audio_array, fs=2048, nperseg=2048, noverlap=3 * 2048 // 4, padded=False)

# Jogamos fora a frequencia de Nyquist
espectro = espectro[:-1]
f = f[:-1]

# Decompoe o espectro em magnitude e fase (angulo).
amplitude_espectro = np.abs(espectro)
amplitude_espectro[amplitude_espectro == 0] = 1e-6
magnitude_espectro = np.log10(np.abs(amplitude_espectro))
phase_espectro = np.angle(espectro)

espectro = polar_to_rect(10 ** magnitude_espectro, phase_espectro)

t, x = signal.istft(Zxx=espectro, fs=2046, nperseg=2046, noverlap=3 * 2046 // 4, )

x = x.astype(np.int16)

sd.play(x, samplerate=16000, blocking=True)
