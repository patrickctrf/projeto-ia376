import csv

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import *
from losses import *
from models import *
from ptk.utils import DataManager


def experiment(device=torch.device("cpu")):
    batch_size = 16
    noise_length = 1
    target_length = 64000
    use_amp = False
    max_examples = 1_000_000

    # Models
    generator = DummyGenerator(noise_length=noise_length, target_length=target_length, n_input_channels=32, n_output_channels=1, kernel_size=7, stride=1, padding=0, dilation=1, bias=True)
    discriminator = Discriminator2D(seq_length=target_length, n_input_channels=25, kernel_size=7, stride=1, padding=0, dilation=1, bias=True)

    # Put in GPU (if available)
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=1e+10, )
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, )
    generator_scaler = GradScaler()
    discriminator_scaler = GradScaler()

    # Variable LR
    generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, lambda epoch: max(1 / (1 + 9 * epoch / 75000), 0.1))
    discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(discriminator_optimizer, lambda epoch: max(1 / (1 + 9 * epoch / 75000), 0.1))

    # loss
    loss = HyperbolicLoss()

    # Train Data
    train_dataset = NsynthDatasetFourier(path="/media/patrickctrf/1226468E26467331/Users/patri/3D Objects/projeto-ia376/E2/nsynth-train/", noise_length=noise_length)
    train_dataset = Subset(train_dataset, [0, ])  # dummy dataset for testing script
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Log data
    total_generator_loss = best_loss = 9999
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

        # Easy concurrent conversion of tensor's device and type
        train_datamanager = DataManager(train_dataloader, device=device, buffer_size=3)
        for x_train, y_train in train_datamanager:
            true_labels = torch.ones((x_train.shape[0], 1), device=device)
            fake_labels = torch.zeros((x_train.shape[0], 1), device=device)

            # Train the generator
            generator_optimizer.zero_grad()
            with autocast(enabled=use_amp):
                generated_data = generator(x_train)

                generator_discriminator_out = discriminator(torch.cat((generated_data, y_train[:, 2:, ]), dim=1))
                generator_loss = loss(generator_discriminator_out, true_labels)

            generator_scaler.scale(generator_loss).backward()
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()

            # Train the discriminator on the true/generated data
            discriminator_optimizer.zero_grad()
            with autocast(enabled=use_amp):
                true_discriminator_out = discriminator(y_train)
                true_discriminator_loss = loss(true_discriminator_out, true_labels)

                # .detach() here, important
                generator_discriminator_out = discriminator(torch.cat((generated_data.detach(), y_train[:, 2:, ]), dim=1))
                generator_discriminator_loss = loss(generator_discriminator_out, fake_labels)

                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2

            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            # LR scheduler update
            discriminator_scheduler.step()
            generator_scheduler.step()

            # Update number of examples seen with current mini batch
            n_examples += x_train.shape[0]

            # Logs
            tqdm_bar_epoch.set_description(
                f'current_generator_loss: {total_generator_loss:5.5f}' +
                f'. disc_fake_err: {generator_discriminator_out.detach().mean():5.5f}' +
                f'. disc_real_acc: {true_discriminator_out.detach().mean():5.5f}' +
                f'. gen_lr: {get_lr(generator_optimizer):1.6f}' +
                f'. disc_lr: {get_lr(discriminator_optimizer):1.6f}'
            )
            tqdm_bar_epoch.update(x_train.shape[0])
            w.writerow([n_examples, total_generator_loss])
            f.flush()

            # Moving average of current model loss
            total_generator_loss = 0.9 * total_generator_loss + 0.1 * generator_loss.detach().item()

            # Checkpoint to best models found.
            if n_examples > last_checkpoint + 1 * batch_size and best_loss > total_generator_loss:
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
