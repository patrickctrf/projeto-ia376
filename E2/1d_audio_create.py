import torch
import sounddevice as sd


def unscale_data(array, means=0.0, stds=3000.0):
    return array * stds + means


generator = torch.load("generator.pth").to(torch.device("cpu"))

ruido_e_classes = torch.cat(
    (
        torch.normal(mean=0, std=1.0, size=(9, 1)),
        torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ]).view(-1, 1),
        torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).view(-1, 1),
    ), dim=0).view(1, 32, 1)

audio_sintetizado = generator(ruido_e_classes).view(-1)

sd.play(unscale_data(audio_sintetizado.detach().numpy(), means=0.0, stds=300.0), samplerate=16000, blocking=True)
