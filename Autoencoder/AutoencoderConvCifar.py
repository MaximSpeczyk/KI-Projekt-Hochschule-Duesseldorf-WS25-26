import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os

# Erstellung der notwendigen Ausgabeverzeichnisse
os.makedirs("./ae_visuals", exist_ok=True)
os.makedirs("./ae_latents", exist_ok=True)


# Definition der Convolutional Autoencoder Architektur
class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=128):
        super().__init__()
        # Initialisierung der Encoder-Schichten (Feature Extraction)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        # Projektion der Features auf den Latent Space (Bottleneck)
        # Berechnung: 128 Kanäle * 4x4 Ortsauflösung = 2048
        self.fc_enc = nn.Linear(128 * 4 * 4, bottleneck_dim)

        # Initialisierung der Decoder-Schichten (Rekonstruktion)
        self.fc_dec = nn.Linear(bottleneck_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Führt den kompletten Forward-Pass (Encode -> Decode) aus."""
        z = self.encoder(x)
        z = self.flatten(z)
        z = self.fc_enc(z)
        out = self.fc_dec(z)
        out = out.view(-1, 128, 4, 4)
        out = self.decoder(out)
        return out

    def encode(self, x):
        """Extrahiert den Latent Vector ohne Dekodierung."""
        with torch.no_grad():
            z = self.encoder(x)
            z = self.flatten(z)
            z = self.fc_enc(z)
            return z


# Start des Trainingsprozesses
if __name__ == "__main__":
    # Konfiguration der Hyperparameter
    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 1e-3

    # Auswahl der Hardware (GPU falls verfügbar)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start Training auf: {device} | Epochen: {EPOCHS}")

    # Laden und Transformieren des CIFAR-10 Datensatzes
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Initialisierung von Modell, Verlustfunktion und Optimierer
    model = ConvAutoencoder(bottleneck_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Liste zur Speicherung des Loss-Verlaufs
    loss_history = []

    # Ausführung der Trainingsschleife
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for imgs, _ in loader:
            imgs = imgs.to(device).float()

            # Vorwärtsdurchlauf
            outputs = model(imgs)
            loss = criterion(outputs, imgs)

            # Rückwärtsdurchlauf und Gewichtsanpassung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Berechnung des Durchschnittsverlusts
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoche {epoch + 1}/{EPOCHS}, Loss={avg_loss:.6f}")

    # 1. Visualisierung: Erstellung des Loss-Verlaufsdiagramms
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.title("Autoencoder Training Loss (CIFAR-10)")
    plt.xlabel("Epoche")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("./ae_visuals/training_loss.png")
    plt.close()
    print("Loss-Diagramm gespeichert in ./ae_visuals/training_loss.png")

    # 2. Visualisierung: Vergleich von Original und Rekonstruktion
    model.eval()
    with torch.no_grad():
        # Laden eines Batches
        dataiter = iter(loader)
        images, _ = next(dataiter)
        images = images.to(device)

        # Generierung der Rekonstruktion
        outputs = model(images)

        # Auswahl der ersten 8 Bilder
        orig_imgs = images[:8]
        recon_imgs = outputs[:8]

        # Erstellung des Vergleichsgitters
        comparison = torch.cat([orig_imgs, recon_imgs])

        # Speichern des Bildes
        utils.save_image(comparison, "./ae_visuals/reconstruction_sample.png", nrow=8)
        print("Vergleichsbild gespeichert in ./ae_visuals/reconstruction_sample.png")

    # Extraktion der Latent-Space-Repräsentationen
    print("Starte Extraktion der Features...")
    dataloader_full = DataLoader(dataset, batch_size=256, shuffle=False)
    latents, labels = [], []

    with torch.no_grad():
        for imgs, labs in dataloader_full:
            imgs = imgs.to(device).float()
            z = model.encode(imgs)
            latents.append(z.cpu().numpy())
            labels.append(labs.numpy())

    # Zusammenfügen der Arrays
    latents = np.vstack(latents)
    labels = np.hstack(labels)

    # Speichern der Modelle und Daten
    torch.save(model.state_dict(), "./ae_latents/cifar_autoencoder.pth")
    print("Modell-Gewichte gespeichert.")
    np.save("./ae_latents/cifar_conv_latents.npy", latents)
    np.save("./ae_latents/cifar_labels.npy", labels)
    print(f"Verarbeitung abgeschlossen. Latents Shape: {latents.shape}")