import numpy as np
import torch
import sounddevice as sd
from scipy import signal


def polar_to_rect(amplitude, angle):
    return amplitude * (np.cos(angle) + 1j * np.sin(angle))


generator = torch.load("checkpoints/best_generator.pth", map_location=torch.device("cpu")).train()

# ruido_e_classes = torch.cat(
#     (
#         torch.normal(mean=0, std=1.0, size=(9, 1)),
#         torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ]).view(-1, 1),
#         torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).view(-1, 1),
#     ), dim=0).view(1, 32, 1, 1)

ruido_e_classes = torch.normal(mean=0, std=1.0, size=(128, 16)).view(1, 128, 16)

fourier_sintetizado = generator(ruido_e_classes)[0].detach().numpy()

espectro = polar_to_rect(10 ** fourier_sintetizado[0], fourier_sintetizado[1])

t, x = signal.istft(Zxx=espectro, fs=2046, nperseg=2046, noverlap=3 * 2046 // 4, )

x = x.astype(np.int16)

sd.play(x, samplerate=16000, blocking=True)
