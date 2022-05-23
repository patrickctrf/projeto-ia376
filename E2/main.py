import csv
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss, MSELoss
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import *
from datasets import *
from models import *
from ptk.utils import DataManager


def experiment(device=torch.device("cpu")):
    epochs = 10
    batch_size = 16
    noise_length = 1
    target_length = 64000
    use_amp = True

    # Models
    generator = Generator2DUpsampled(noise_length=noise_length, target_length=target_length, n_input_channels=32, n_output_channels=1, kernel_size=7, stride=1, padding=0, dilation=1, bias=True)
    discriminator = Discriminator2D(seq_length=target_length, n_input_channels=25, kernel_size=7, stride=1, padding=0, dilation=1, bias=True)

    # Put in GPU (if available)
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.1, )
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.01, )
    generator_scaler = GradScaler()
    discriminator_scaler = GradScaler()

    # loss
    loss = MSELoss()

    # Train Data
    train_dataset = NsynthDatasetFourier(path="/media/patrickctrf/1226468E26467331/Users/patri/3D Objects/projeto-ia376/E2/nsynth-train/", noise_length=noise_length)
    # Carrega os dados em mini batches, evita memory overflow
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # Log data
    best_validation_loss = 999999999
    f = open("loss_log.csv", "w")
    w = csv.writer(f)
    w.writerow(["epoch", "training_loss"])
    tqdm_bar_epoch = tqdm(range(epochs))
    tqdm_bar_epoch.set_description("epoch: 0. ")

    discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discriminator_optimizer, T_max=5000, eta_min=0.00001)
    set_lr(discriminator_optimizer, new_lr=0.01)

    for i in tqdm_bar_epoch:
        total_generator_loss = 0

        generator.train()
        discriminator.train()

        # Variable LR. Restart every epoch
        generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optimizer, milestones=[1500, 3500, 15000, ], gamma=0.1)  # 25000
        # discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=0.99)
        set_lr(generator_optimizer, new_lr=0.1)

        # Facilita e acelera a transferência de dispositivos (Cpu/GPU)
        train_datamanager = DataManager(train_dataloader, device=device, buffer_size=1)

        tqdm_bar_iter = tqdm(train_datamanager, total=len(train_dataloader))
        for x_train, y_train in tqdm_bar_iter:
            # Comodidade para dizer que as saidas sao verdadeiras ou falsas
            true_labels = torch.ones((x_train.shape[0], 1), device=device)
            fake_labels = torch.zeros((x_train.shape[0], 1), device=device)

            # t0 = time.time()

            # zero the gradients on each iteration
            generator_optimizer.zero_grad()

            with autocast(enabled=use_amp):
                generated_data = generator(x_train)

                # print("generator_output: ", time.time() - t0)
                # t0 = time.time()

                # Train the generator
                # We invert the labels here and don't train the discriminator because we want the generator
                # to make things the discriminator classifies as true.
                generator_discriminator_out = discriminator(torch.cat((generated_data, y_train[:, 2:, ]), dim=1))
                generator_loss = loss(generator_discriminator_out, true_labels)

            generator_scaler.scale(generator_loss).backward()
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()

            # just free memory
            generator_loss = generator_loss.detach()
            generated_data = generated_data.detach()

            # print("generator_backward: ", time.time() - t0)
            # t0 = time.time()

            # Train the discriminator on the true/generated data
            discriminator_optimizer.zero_grad()

            with autocast(enabled=use_amp):
                true_discriminator_out = discriminator(y_train)
                true_discriminator_loss = loss(true_discriminator_out, true_labels)

                # print("discriminator_output: ", time.time() - t0)
                # t0 = time.time()

                # add .detach() here think about this
                generator_discriminator_out = discriminator(torch.cat((generated_data.detach(), y_train[:, 2:, ]), dim=1))
                generator_discriminator_loss = loss(generator_discriminator_out, fake_labels)

                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2

            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            # just free memory
            generator_discriminator_out = generator_discriminator_out.detach()
            generator_discriminator_loss = generator_discriminator_loss.detach()
            true_discriminator_out = true_discriminator_out.detach()
            true_discriminator_loss = true_discriminator_loss.detach()
            discriminator_loss = discriminator_loss.detach()

            # print("discriminator_backward: ", time.time() - t0)

            # LR scheduler update
            discriminator_scheduler.step()
            generator_scheduler.step()

            tqdm_bar_iter.set_description(
                f'mini-batch generator_loss: {generator_loss.detach().item():5.5f}' +
                f'. discriminator_gen: {generator_discriminator_out.detach().mean():5.5f}' +
                f'. discriminator_real: {true_discriminator_out.detach().mean():5.5f}'
            )
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
            torch.save(discriminator, "best_discriminator.pth")
            torch.save(discriminator.state_dict(), "best_discriminator_state_dict.pth")

        # Save everything after each epoch
        generator.eval()
        torch.save(generator, "generator.pth")
        torch.save(generator.state_dict(), "generator_state_dict.pth")
        discriminator.eval()
        torch.save(discriminator, "discriminator.pth")
        torch.save(discriminator.state_dict(), "discriminator_state_dict.pth")
    f.close()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, new_lr=0.01):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        return


if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    experiment(device)
