"""
dcgan_train_and_generate.py
"""
# ------ Training MNIST ------
# python dcgan_train_and_generate.py --dataset mnist --epochs 50 --batch-size 128

# ------ Generierung MINST ------
# python dcgan_train_and_generate.py --dataset mnist --generate 10000 --checkpoint ./checkpoints/generator_mnist.pth

# ------ Training CIFAR10 ------
# python dcgan_train_and_generate.py --dataset cifar10 --epochs 100 --batch-size 128

# ------ Generierung CIFAR10 ------
# python dcgan_train_and_generate.py --dataset cifar10 --generate 10000 --checkpoint ./checkpoints/generator_cifar10.pth

import os
import argparse
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

# ---------- Konfiguration & Hyperparameter ----------
LATENT_DIM = 100
NGF = 64  # Basis-Filteranzahl Generator
NDF = 64  # Basis-Filteranzahl Diskriminator
IMG_SIZE = 32
CHECKPOINT_DIR = "./checkpoints"
SAMPLES_DIR = "./generated"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)


# ---------- Gewichtsinitialisierung ----------
def weights_init(m):
    """Initialisierung der Layer-Gewichte nach dem DCGAN-Standard (Normalverteilung)."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------- Modell-Definitionen ----------
class Generator(nn.Module):
    def __init__(self, nz=LATENT_DIM, ngf=NGF, nc=3):
        super().__init__()
        # Transponierte Faltungen zur schrittweisen Hochskalierung des latenten Vektors
        self.main = nn.Sequential(
            # Input Z: (nz) -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # Status: (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Status: (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Status: (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Finale Schicht zur Erzeugung der Bildkanäle (RGB oder Graustufen)
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()  # Normalisiert Output auf den Bereich [-1, 1]
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=NDF):
        super().__init__()
        # Faltungsschichten zur Klassifizierung (Downsampling ohne Pooling)
        self.main = nn.Sequential(
            # Input: nc x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Binärer Output: Wahrscheinlichkeit für 'Echt'
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)


# ---------- Training und Datenprozessierung ----------
def get_dataloader(dataset_name, batch_size):
    """Konfiguration der Datasets und Normalisierung der Pixelwerte."""
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        nc = 1
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        nc = 3
    else:
        raise ValueError("Dataset muss 'mnist' oder 'cifar10' sein")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return loader, nc


def train_dcgan(dataset="cifar10", epochs=50, batch_size=128, device="cpu", save_every=None):
    """Haupt-Trainingsschleife für Generator und Diskriminator."""
    loader, nc = get_dataloader(dataset, batch_size)
    device = torch.device(device)

    netG = Generator(nz=LATENT_DIM, ngf=NGF, nc=nc).to(device)
    netD = Discriminator(nc=nc, ndf=NDF).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Fixes Rauschen zur visuellen Verfolgung des Fortschritts
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for real in pbar:
            imgs = real[0].to(device)
            b_size = imgs.size(0)

            real_label = torch.full((b_size,), 1., dtype=torch.float, device=device)
            fake_label = torch.full((b_size,), 0., dtype=torch.float, device=device)

            # --- Diskriminator-Training ---
            netD.zero_grad()
            output_real = netD(imgs)
            lossD_real = criterion(output_real, real_label)

            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            lossD_fake = criterion(output_fake, fake_label)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # --- Generator-Training ---
            netG.zero_grad()
            output = netD(fake)
            # Ziel: Diskriminator dazu bringen, Fake als Real (1) zu klassifizieren
            lossG = criterion(output, real_label)
            lossG.backward()
            optimizerG.step()

            pbar.set_postfix({"lossD": lossD.item(), "lossG": lossG.item()})

        # Sicherung des aktuellen Modellzustands
        torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_{dataset}.pth"))
        torch.save(netD.state_dict(), os.path.join(CHECKPOINT_DIR, f"discriminator_{dataset}.pth"))

        # Erstellung von Beispiel-Grids zur Validierung
        with torch.no_grad():
            fake_samples = netG(fixed_noise).cpu()
            grid = utils.make_grid(fake_samples, nrow=8, normalize=True, scale_each=True)
            utils.save_image(grid, os.path.join(SAMPLES_DIR, f"{dataset}_epoch{epoch:03d}.png"))

    return os.path.join(CHECKPOINT_DIR, f"generator_{dataset}.pth")


def generate_images_from_checkpoint(checkpoint_path, dataset="cifar10", n_samples=1000, device="cpu", seed=42,
                                    out_dir=SAMPLES_DIR):
    """Inferenz-Funktion zur Erzeugung neuer Bilder aus einem gespeicherten Modell."""
    device = torch.device(device)
    nc = 1 if dataset == "mnist" else 3
    netG = Generator(nz=LATENT_DIM, ngf=NGF, nc=nc).to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()

    torch.manual_seed(seed)
    batch = 256
    n_batches = math.ceil(n_samples / batch)
    images = []
    latents = []

    for i in range(n_batches):
        b = min(batch, n_samples - i * batch)
        z = torch.randn(b, LATENT_DIM, 1, 1, device=device)
        with torch.no_grad():
            gen = netG(z).cpu()

        # Denormalisierung von [-1, 1] auf [0, 1] für die Speicherung
        imgs = (gen + 1.0) / 2.0
        imgs = torch.clamp(imgs, 0.0, 1.0)
        images.append(imgs.numpy())
        latents.append(z.cpu().view(b, LATENT_DIM).numpy())

    images = np.concatenate(images, axis=0)[:n_samples]
    latents = np.concatenate(latents, axis=0)[:n_samples]

    # Speicherung der Rohdaten als NumPy-Arrays
    np.save(os.path.join(out_dir, f"{dataset}_gan_images_{n_samples}.npy"), images)
    np.save(os.path.join(out_dir, f"{dataset}_gan_latents_{n_samples}.npy"), latents)

    # Erstellung einer finalen Übersichtsgrafik (Grid)
    timgs = torch.from_numpy(images).float()
    if nc == 1:
        preview = utils.make_grid(timgs[:64], nrow=8, normalize=False, pad_value=1.0)
    else:
        preview = utils.make_grid(timgs[:64], nrow=8, normalize=False)
    utils.save_image(preview, os.path.join(out_dir, f"{dataset}_gan_preview_{n_samples}.png"))

    return os.path.join(out_dir, f"{dataset}_gan_images_{n_samples}.npy"), os.path.join(out_dir,
                                                                                        f"{dataset}_gan_latents_{n_samples}.npy")


# ---------- Command Line Interface ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--generate", type=int, default=0, help="Anzahl zu generierender Bilder")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pfad zum Generator-Checkpoint")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.generate > 0:
        ckpt = args.checkpoint if args.checkpoint else os.path.join(CHECKPOINT_DIR, f"generator_{args.dataset}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt}")
        generate_images_from_checkpoint(ckpt, dataset=args.dataset, n_samples=args.generate, device=device)
    else:
        train_dcgan(dataset=args.dataset, epochs=args.epochs, batch_size=args.batch_size, device=device)