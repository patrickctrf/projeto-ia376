import csv

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import NsynthDatasetTimeSeries
from models import Discriminator1D, Generator1DUpsampled
from ptk.utils import DataManager


def experiment(device=torch.device("cpu")):
    epochs = 10
    batch_size = 1
    noise_length = 1
    target_length = 64000
    use_amp = True

    # Models
    generator = Generator1DUpsampled(noise_length=noise_length, target_length=target_length, n_input_channels=32, n_output_channels=1, kernel_size=7, stride=1, padding=0, dilation=1)
    discriminator = Discriminator1D(seq_length=target_length, n_input_channels=1, n_output_channels=64, kernel_size=7, stride=1, padding=0, dilation=1)

    # Put in GPU (if available)
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.01)
    generator_scaler = GradScaler()
    discriminator_scaler = GradScaler()

    # loss
    loss = BCEWithLogitsLoss()

    # Train Data
    train_dataset = NsynthDatasetTimeSeries(path="nsynth-train/", noise_length=noise_length)
    # Carrega os dados em mini batches, evita memory overflow
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Facilita e acelera a transferência de dispositivos (Cpu/GPU)
    train_datamanager = DataManager(train_dataloader, device=device, buffer_size=1)

    # # Validation Data
    # valid_dataset = NsynthDatasetTimeSeries(path="nsynth-valid/", noise_length=noise_length)
    # # O tamanho do mini batch de validacao tem que ser tal que o dataloader de
    # # validacao tenho o mesmo tamanho do de treino
    # validation_batch_size = len(valid_dataset) // len(train_dataloader)
    # assert validation_batch_size > 0, 'Train dataloader is bigger than validation dataset'
    # valid_dataloader = DataLoader(valid_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # # Facilita e acelera a transferência de dispositivos (Cpu/GPU)
    # valid_datamanager = DataManager(valid_dataloader, device=device, buffer_size=1)

    best_validation_loss = 999999999
    f = open("loss_log.csv", "w")
    w = csv.writer(f)
    w.writerow(["epoch", "training_loss"])
    tqdm_bar_epoch = tqdm(range(epochs))
    tqdm_bar_iter = tqdm(train_datamanager, total=len(train_dataloader))
    tqdm_bar_epoch.set_description("epoch: 0. ")
    for i in tqdm_bar_epoch:
        total_generator_loss = 0
        generator.train()
        discriminator.train()
        # for (x_train, y_train), (x_valid, y_valid) in tqdm(zip(train_datamanager, valid_datamanager), total=len(train_dataloader)):
        for x_train, y_train in tqdm_bar_iter:
            # Comodidade para dizer que as saidas sao verdadeiras ou falsas
            true_labels = torch.ones((x_train.shape[0], 1), device=device)
            fake_labels = torch.zeros((x_train.shape[0], 1), device=device)

            # zero the gradients on each iteration
            generator_optimizer.zero_grad()
            with autocast(use_amp):
                generated_data = generator(x_train)

                # print("check0")

                # Train the generator
                # We invert the labels here and don't train the discriminator because we want the generator
                # to make things the discriminator classifies as true.
                generator_discriminator_out = discriminator(generated_data)
                generator_loss = loss(generator_discriminator_out, true_labels)
            generator_scaler.scale(generator_loss).backward()
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()

            # print("check1")

            # Train the discriminator on the true/generated data
            discriminator_optimizer.zero_grad()
            with autocast(use_amp):
                true_discriminator_out = discriminator(y_train)  # x_train[:, 0:1])
                true_discriminator_loss = loss(true_discriminator_out, true_labels)

                # print("check2")

                # add .detach() here think about this
                generator_discriminator_out = discriminator(generated_data.detach())
                generator_discriminator_loss = loss(generator_discriminator_out, fake_labels)
                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            # print("check3")

            tqdm_bar_iter.set_description(f'mini-batch generator_loss: {generator_loss.detach().item():15.15f}')
            total_generator_loss += generator_loss.detach().item()

        total_generator_loss /= len(train_dataloader)
        tqdm_bar_epoch.set_description(f'epoch: {i:1} generator_loss: {total_generator_loss:15.15f}')
        w.writerow([i, total_generator_loss])
        f.flush()

        # Checkpoint to best models found.
        if best_validation_loss > total_generator_loss:
            # Update the new best loss.
            best_validation_loss = total_generator_loss
            generator.eval()
            torch.save(generator, "best_generator.pth")
            torch.save(generator.state_dict(), "best_generator_state_dict.pth")
            discriminator.eval()
            torch.save(generator, "best_discriminator.pth")
            torch.save(generator.state_dict(), "best_discriminator_state_dict.pth")
    f.close()


if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    experiment(device)
