import csv

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import HyperbolicLoss
from datasets import *
from models import *
from models import TransformerGenerator
from ptk.utils import DataManager


def experiment(device=torch.device("cpu")):
    batch_size = 16
    noise_length = 16
    target_length = 128
    use_amp = True
    max_examples = 1_000_000

    # Models
    generator = TransformerGenerator(dim=2048, input_size=noise_length, output_size=2048, max_seq_length=target_length, n_layers=10)
    # discriminator = TransformerDiscriminator(dim=256, input_size=target_length, output_size=1, max_seq_length=target_length)
    discriminator = Discriminator2D(seq_length=target_length, n_input_channels=2, kernel_size=7, stride=1, padding=0, dilation=1, bias=True)

    # Put in GPU (if available)
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3, )
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3, )
    generator_scaler = GradScaler()
    discriminator_scaler = GradScaler()

    # Variable LR
    generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, lambda epoch: max(1 - epoch / 30000.0, 0.1))
    discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(discriminator_optimizer, lambda epoch: max(1 - epoch / 30000.0, 0.1))

    # loss
    criterion = MSELoss()

    # Train Data
    train_dataset = NsynthDatasetFourier(path="/media/patrickctrf/1226468E26467331/Users/patri/3D Objects/projeto-ia376/E2/nsynth-train/", noise_length=noise_length)
    # train_dataset = Subset(train_dataset, [0, ])  # dummy dataset for testing script
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Log data
    total_generator_loss = best_loss = 9999.0
    f = open("loss_log.csv", "w")
    w = csv.writer(f)
    w.writerow(["epoch", "training_loss"])
    tqdm_bar_epoch = tqdm(range(max_examples))
    tqdm_bar_epoch.set_description("epoch: 0. ")
    last_checkpoint = 0

    n_examples = 0
    while n_examples < max_examples:
        generator.train()
        discriminator.train()

        # Facilita e acelera a transferência de dispositivos (Cpu/GPU)
        train_datamanager = DataManager(train_dataloader, device=device, buffer_size=3)
        for x_train, y_train in train_datamanager:
            # Comodidade para dizer que as saidas sao verdadeiras ou falsas
            true_labels = torch.ones((x_train.shape[0], 1), device=device)
            fake_labels = torch.zeros((x_train.shape[0], 1), device=device)

            # zero the gradients on each iteration
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            with autocast(enabled=use_amp):
                generated_data = generator(x_train)

                # Train the generator
                # We invert the labels here and don't train the discriminator because we want the generator
                # to make things the discriminator classifies as true.
                generator_discriminator_out = discriminator(generated_data)
                # generator_loss = criterion(generated_data, y_train[:, :2, ])
                generator_loss = criterion(generator_discriminator_out, true_labels)

            generator_scaler.scale(generator_loss).backward()
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()

            # Train the discriminator on the true/generated data
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            with autocast(enabled=use_amp):
                true_discriminator_out = discriminator(y_train)
                true_discriminator_loss = criterion(true_discriminator_out, true_labels)

                # add .detach() here think about this
                generator_discriminator_out = discriminator(generated_data.detach())
                generator_discriminator_loss = criterion(generator_discriminator_out, fake_labels)

                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2

            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            # LR scheduler update
            discriminator_scheduler.step()
            generator_scheduler.step()

            tqdm_bar_epoch.set_description(
                f'current_generator_loss: {total_generator_loss:5.5f}' +
                f'. disc_fake_err: {generator_discriminator_out.detach().mean():5.5f}' +
                f'. disc_real_acc: {true_discriminator_out.detach().mean():5.5f}' +
                f'. gen_lr: {get_lr(generator_optimizer):1.6f}' +
                f'. disc_lr: {get_lr(discriminator_optimizer):1.6f}'
            )
            tqdm_bar_epoch.update(x_train.shape[0])

            n_examples += x_train.shape[0]

            w.writerow([n_examples, total_generator_loss])
            f.flush()

            total_generator_loss = 0.9 * total_generator_loss + 0.1 * generator_loss.detach().item()

            # Checkpoint to best models found.
            if n_examples > last_checkpoint + 100 * batch_size and best_loss > total_generator_loss:
                # Update the new best loss.
                best_loss = total_generator_loss
                last_checkpoint = n_examples
                generator.eval()
                torch.save(generator, "checkpoints/best_generator.pth")
                torch.save(generator.state_dict(), "checkpoints/best_generator_state_dict.pth")
                discriminator.eval()
                torch.save(discriminator, "checkpoints/best_discriminator.pth")
                torch.save(discriminator.state_dict(), "checkpoints/best_discriminator_state_dict.pth")
                print("\ncheckpoint!\n")
                generator.train()
                discriminator.train()

        # Save everything after each epoch
        generator.eval()
        torch.save(generator, "checkpoints/generator.pth")
        torch.save(generator.state_dict(), "checkpoints/generator_state_dict.pth")
        discriminator.eval()
        torch.save(discriminator, "checkpoints/discriminator.pth")
        torch.save(discriminator.state_dict(), "checkpoints/discriminator_state_dict.pth")
        generator.train()
        discriminator.train()
    f.close()


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


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
