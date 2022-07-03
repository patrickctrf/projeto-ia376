import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ptk.utils import DataManager

__all__ = ["NsynthDatasetTimeSeries", "NsynthDatasetFourier"]


class NsynthDatasetFourier(Dataset):

    def __init__(self, path="nsynth-train/", noise_length=256, shuffle=True):
        super().__init__()
        self.noise_length = noise_length
        self.path = path

        instr_fmly_name = ["bass", "brass", "flute", "guitar", "keyboard",
                           "mallet", "organ", "reed", "string", "synth_lead",
                           "vocal"]
        instr_fmly_num = range(len(instr_fmly_name))  # 11 familias

        self.instr_fmly_dict = dict(zip(instr_fmly_name, instr_fmly_num))

        notas_nomes = ["do", "do_s", "re", "re_s", "mi", "fa", "fa_s", "sol",
                       "sol_s", "la", "la_s", "si"]
        self.notas_indices = range(len(notas_nomes))  # 12 seminotas

        self.notas_dict = dict(zip(notas_nomes, self.notas_indices))

        with open(os.path.join(path, 'examples.json'), 'r') as file:
            summary_dict = json.load(file)

        self.summary_df = pd.DataFrame(list(summary_dict.values()))
        # self.summary_df = pd.get_dummies(self.summary_df, columns=["instrument_family_str", ])

        self.shuffle_array = torch.arange(self.summary_df.shape[0])
        if shuffle is True: self.shuffle_array = torch.randperm(self.summary_df.shape[0])

    def __getitem__(self, index):
        idx = self.shuffle_array[index].item()

        sample_info = self.summary_df.loc[idx]

        sample_audio_array = wavfile.read(os.path.join(self.path, "audio", sample_info["note_str"]) + ".wav")[1] / 1.0
        # sample_audio_array = _scale_data(sample_audio_array)
        sample_audio_array = np.pad(sample_audio_array, [(700, 700), ], mode='constant')

        f, t, espectro = signal.stft(x=sample_audio_array, fs=2048, nperseg=2048, noverlap=3 * 2048 // 4, padded=False)

        # espectro = tf.signal.stft(
        #     signals=sample_audio_array,
        #     frame_length=2048,
        #     frame_step=512,
        #     fft_length=2048,
        #     pad_end=10,
        # )
        #
        # espectro = espectro.numpy().T

        # espectro = torch.stft(
        #     input=torch.tensor(sample_audio_array.copy()),
        #     n_fft=2048,
        #     win_length=2048,
        #     hop_length=2048 // 4,
        #     return_complex=True,
        #     center=True,
        # ).numpy()

        # Jogamos fora a frequencia de Nyquist
        espectro = espectro[:-1]
        f = f[:-1]

        # # Visualizar o espectro gerado
        # plt.pcolormesh(t, f, np.abs(espectro), vmin=0, vmax=1e5, shading='gouraud')
        # plt.show()

        instr_fmly_one_hot = torch.zeros(((len(self.instr_fmly_dict.keys()),) + espectro.shape))
        notas_one_hot = torch.zeros(((len(self.notas_indices),) + espectro.shape))

        # Decompoe o espectro em magnitude e fase (angulo).
        amplitude_espectro = np.abs(espectro)
        amplitude_espectro[amplitude_espectro == 0] = 1e-6
        magnitude_espectro = np.log10(amplitude_espectro)
        phase_espectro = np.angle(espectro)

        # Setamos como 1 o elemento daquela familia (one hot)
        instr_fmly_one_hot[self.instr_fmly_dict[sample_info["instrument_family_str"]]] = 1
        # pitch dividido por 12 (qtde de notas), para transcrever em nota
        # dentro da "oitava" em que aquele pitch se encontra.
        notas_one_hot[self.notas_indices[sample_info["pitch"] % len(self.notas_indices)]] = 1

        # Retorna X e Y:
        # Sendo x == (ruido, one hot do timbre (family), one hot das notas)
        # # Sendo y == (sample_de_audio, one hot do timbre (family), one hot das notas)
        # return torch.cat((torch.normal(mean=0, std=1.0, size=(9, self.noise_length, self.noise_length)), instr_fmly_one_hot[:, 0:1, 0:1], notas_one_hot[:, 0:1, 0:1]), dim=0).detach(), \
        #        torch.cat((torch.tensor(magnitude_espectro[np.newaxis, ...]), torch.tensor(phase_espectro[np.newaxis, ...]), instr_fmly_one_hot, notas_one_hot), dim=0).detach()

        return torch.normal(mean=0, std=1.0, size=(256, self.noise_length, self.noise_length)), \
               torch.cat((torch.tensor(magnitude_espectro[np.newaxis, ...]), torch.tensor(phase_espectro[np.newaxis, ...]),), dim=0)

    def __len__(self, ):
        return self.summary_df.shape[0]


def _scale_data(array, means=0.0, stds=32768.0 / 2):
    return (array - means) / stds


def _unscale_data(array, means=0.0, stds=32768.0 / 2):
    return array * stds + means


class NsynthDatasetTimeSeries(Dataset):

    def __init__(self, path="nsynth-train/", noise_length=256, shuffle=True):
        super().__init__()
        self.noise_length = noise_length
        self.path = path

        instr_fmly_name = ["bass", "brass", "flute", "guitar", "keyboard",
                           "mallet", "organ", "reed", "string", "synth_lead",
                           "vocal"]
        instr_fmly_num = range(len(instr_fmly_name))  # 11 familias

        self.instr_fmly_dict = dict(zip(instr_fmly_name, instr_fmly_num))

        notas_nomes = ["do", "do_s", "re", "re_s", "mi", "fa", "fa_s", "sol",
                       "sol_s", "la", "la_s", "si"]
        self.notas_indices = range(len(notas_nomes))  # 12 seminotas

        self.notas_dict = dict(zip(notas_nomes, self.notas_indices))

        with open(os.path.join(path, 'examples.json'), 'r') as file:
            summary_dict = json.load(file)

        self.summary_df = pd.DataFrame(list(summary_dict.values()))
        # self.summary_df = pd.get_dummies(self.summary_df, columns=["instrument_family_str", ])

        self.shuffle_array = torch.arange(self.summary_df.shape[0])
        if shuffle is True: self.shuffle_array = torch.randperm(self.summary_df.shape[0])

    def __getitem__(self, index):
        idx = self.shuffle_array[index].item()

        sample_info = self.summary_df.loc[idx]

        sample_audio_array = wavfile.read(os.path.join(self.path, "audio", sample_info["note_str"]) + ".wav")[1]
        sample_audio_array = _scale_data(sample_audio_array.reshape(1, -1))

        instr_fmly_one_hot = torch.zeros((len(self.instr_fmly_dict.keys()), self.noise_length))
        notas_one_hot = torch.zeros((len(self.notas_indices), self.noise_length))

        # Setamos como 1 o elemento daquela familia (one hot)
        instr_fmly_one_hot[self.instr_fmly_dict[sample_info["instrument_family_str"]]] = 1
        # pitch dividido por 12 (qtde de notas), para transcrever em nota
        # dentro da "oitava" em que aquele pitch se encontra.
        notas_one_hot[self.notas_indices[sample_info["pitch"] % len(self.notas_indices)]] = 1

        # Retorna X e Y:
        # Sendo x == (ruido, one hot do timbre (family), one hot das notas)
        # Sendo y == (sample_de_audio,)
        return torch.cat((torch.normal(mean=0, std=1.0, size=(9, self.noise_length)), instr_fmly_one_hot, notas_one_hot), dim=0).detach(), \
               torch.cat((torch.tensor(sample_audio_array), instr_fmly_one_hot.repeat(1, 64000), notas_one_hot.repeat(1, 64000)), dim=0).detach()

    def __len__(self, ):
        return self.summary_df.shape[0]


if __name__ == '__main__':
    # dataset = NsynthDatasetTimeSeries(path="nsynth-train/", shuffle=True)
    #
    # x = dataset[0]
    #
    # dataloader = DataLoader(dataset, batch_size=256, shuffle=True, )  # num_workers=4)

    # Train Data

    epochs = 10
    batch_size = 16
    noise_length = 1
    target_length = 64000

    device = torch.device("cpu")

    train_dataset = NsynthDatasetFourier(path="/media/patrickctrf/1226468E26467331/Users/patri/3D Objects/projeto-ia376/E2/nsynth-train/", noise_length=noise_length, shuffle=False)
    x = train_dataset[0]
    # Carrega os dados em mini batches, evita memory overflow
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # Facilita e acelera a transferência de dispositivos (Cpu/GPU)
    train_datamanager = DataManager(train_dataloader, device=device, buffer_size=30)

    x = 0
    minimo = 0
    maximo = 0
    for entrada, saida in tqdm(train_datamanager):
        # x = x + 1
        maximo = max(maximo, saida.detach().cpu().numpy()[:, 0].max())
        minimo = min(minimo, saida.detach().cpu().numpy()[:, 0].min())

    x = 1
