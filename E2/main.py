import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import NsynthDatasetTimeSeries
from models import Generator1D, Discriminator1D
from ptk.utils import DataManager


def experiment(device=torch.device("cpu")):
    epochs = 20

    model = Generator1D(n_input_channels=24, n_output_channels=64,
                        kernel_size=7, stride=1, padding=0, dilation=1)
    model.to(device)

    dataset = NsynthDatasetTimeSeries(path="nsynth-test/")

    # Carrega os dados em mini batches, evita memory overflow
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    # Facilita e acelera a transferência de dispositivos (Cpu/GPU)
    datamanager = DataManager(dataloader, device=device, buffer_size=3)

    for i in tqdm(range(epochs)):
        for x, y in tqdm(datamanager):
            # print(model(x).shape)
            model(x)


if __name__ == '__main__':
    if 0 and torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    experiment(device)


def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):

    # Models
    generator = Generator1D(input_length)
    discriminator = Discriminator1D(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = BCEWithLogitsLoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
