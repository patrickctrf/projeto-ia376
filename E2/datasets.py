import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class NsynthDataset(Dataset):

    def __init__(self, path="nsynth-valid/", shuffle=True):
        super().__init__()

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

        self.shuffle_array = np.arange(self.summary_df.shape[0])
        if shuffle is True: np.random.shuffle(self.shuffle_array)

    def __getitem__(self, index):
        idx = self.shuffle_array[index]

        sample_info = self.summary_df.loc[idx]

        sample_audio_array = wavfile.read(os.path.join(self.path, "audio", sample_info["note_str"]) + ".wav")[1]
        sample_audio_array = self._scale_data(sample_audio_array.reshape(1, -1))

        intr_fmly_one_hot = torch.zeros((len(self.instr_fmly_dict.keys()), 64000))
        notas_one_hot = torch.zeros((len(self.notas_indices), 64000))

        # Setamos como 1 o elemento daquela familia (one hot)
        intr_fmly_one_hot[self.instr_fmly_dict[sample_info["instrument_family_str"]]] = 1
        # pitch dividido por 12 (qtde de notas), para transcrever em nota
        # dentro da "oitava" em que aquele pitch se encontra.
        notas_one_hot[self.notas_indices[sample_info["pitch"] % len(self.notas_indices)]] = 1

        # Retorna X e Y:
        # Sendo x == (ruido, one hot do timbre (family), one hot das notas)
        # Sendo y == (sample_de_audio, one hot do timbre, one hot das notas)
        return torch.cat((torch.normal(mean=0, std=0.5, size=(1, 64000)), intr_fmly_one_hot, notas_one_hot), dim=0), \
               torch.cat((torch.tensor(sample_audio_array), intr_fmly_one_hot, notas_one_hot), dim=0)

    def __len__(self, ):
        return self.summary_df.shape[0]

    def _scale_data(self, array, means=0.0, stds=30000):
        return (array - means) / stds

    def _unscale_data(self, array, means=0.0, stds=30000):
        return array * stds + means


dataset = NsynthDataset(path="nsynth-train/", shuffle=True)

x = dataset[0]

dataloader = DataLoader(dataset, batch_size=256, shuffle=True, )  # num_workers=4)

for sample in tqdm(dataloader):
    print(sample)

x = 1