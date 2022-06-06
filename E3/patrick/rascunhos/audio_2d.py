import os

import numpy as np
import torch
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import DataLoader

from E3.patrick.datasets import NsynthDatasetFourier


def unscale_data(array, means=0.0, stds=3000.0):
    return array * stds + means


def polar_to_rect(amplitude, angle):
    return amplitude * (np.cos(angle) + 1j * np.sin(angle))


# sample_audio_array = wavfile.read(os.path.join("/home/patrickctrf/Documentos/pode_apagar/projeto-ia376/E2/nsynth-train/audio/guitar_acoustic_000-025-050", ) + ".wav")[1]
# # wavfile.write("/home/patrickctrf/Documentos/pode_apagar/projeto-ia376/E2/nsynth-train/guitar_acoustic_000-025-050.wav", 16000, sample_audio_array)
# sample_audio_array = np.pad(sample_audio_array, [(700, 700), ], mode='constant')
#
# sd.play(sample_audio_array, samplerate=16000, blocking=True)
#
# f, t, espectro = signal.stft(x=sample_audio_array, fs=2048, nperseg=2048, noverlap=3 * 2048 // 4, padded=False)
#
# # Jogamos fora a frequencia de Nyquist
# espectro = espectro[:-1]
# f = f[:-1]
#
# # Decompoe o espectro em magnitude e fase (angulo).
# amplitude_espectro = np.abs(espectro)
# amplitude_espectro[amplitude_espectro == 0] = 1e-6
# magnitude_espectro = np.log10(np.abs(amplitude_espectro))
# phase_espectro = np.angle(espectro)

# Train Data
train_dataset = NsynthDatasetFourier(path="/media/patrickctrf/1226468E26467331/Users/patri/3D Objects/projeto-ia376/E2/nsynth-train/", noise_length=1)
# train_dataset = Subset(train_dataset, [0, 1]) # dummy dataset for testing script
# Carrega os dados em mini batches, evita memory overflow
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

data_loaded = next(iter(train_dataloader))[1][0]

magnitude_espectro = data_loaded[0]
phase_espectro = data_loaded[1]

espectro = polar_to_rect(10 ** magnitude_espectro, phase_espectro)

t, x = signal.istft(Zxx=espectro, fs=2046, nperseg=2046, noverlap=3 * 2046 // 4, )

x = x.astype(np.int16)

sd.play(x, samplerate=16000, blocking=True)
