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


# Definition der Autoencoder-Architektur
class LinearAutoencoder(nn.Module):
    """
    Implementiert einen einfachen linearen Autoencoder für MNIST.
    Struktur: Encoder (Input -> Latent) und Decoder (Latent -> Rekonstruktion).
    """
    def __init__(self, input_dim=784, bottleneck_dim=32):
        super().__init__()
        # Definition des Encoders: Reduktion der Dimensionen
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim)
        )
        # Definition des Decoders: Rekonstruktion der Eingabedaten
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Skalierung der Ausgabe auf den Bereich [0, 1]
        )

    def forward(self, x):
        """Leitet die Eingabe durch Encoder und Decoder."""
        return self.decoder(self.encoder(x))

    def encode(self, x):
        """Extrahiert die Darstellung im Latent Space ohne Gradientenberechnung."""
        with torch.no_grad():
            return self.encoder(x)


# Start des Trainingsprozesses
if __name__ == "__main__":
    # Konfiguration der Hyperparameter
    EPOCHS = 30
    BATCH_SIZE = 128
    LR = 1e-3
    BOTTLENECK = 32

    # Auswahl der Hardware (GPU falls verfügbar, sonst CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start MNIST Training auf: {device} | Epochen: {EPOCHS}")

    # Laden und Transformieren des Datensatzes
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Initialisierung von Modell, Verlustfunktion und Optimierer
    model = LinearAutoencoder(input_dim=784, bottleneck_dim=BOTTLENECK).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_history = []

    # Ausführung der Trainingsschleife
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for imgs, _ in loader:
            # Umformung der Bilddaten (Flatten) und Transfer auf das Gerät
            imgs = imgs.view(imgs.size(0), -1).to(device).float()

            # Vorwärtsdurchlauf und Fehlerberechnung
            outputs = model(imgs)
            loss = criterion(outputs, imgs)

            # Rückwärtsdurchlauf und Aktualisierung der Gewichte
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Berechnung des Durchschnittsverlusts pro Epoche
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoche {epoch + 1}/{EPOCHS}, Loss={avg_loss:.6f}")

    # 1. Visualisierung: Erstellung des Loss-Verlaufsdiagramms
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.title("Linear Autoencoder Training Loss (MNIST)")
    plt.xlabel("Epoche")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("./ae_visuals/mnist_training_loss.png")
    plt.close()
    print("Loss-Diagramm gespeichert: ./ae_visuals/mnist_training_loss.png")

    # 2. Visualisierung: Vergleich von Original und Rekonstruktion
    model.eval()
    with torch.no_grad():
        # Laden eines Batches zur Visualisierung
        dataiter = iter(loader)
        images_original, _ = next(dataiter)
        images_flat = images_original.view(images_original.size(0), -1).to(device)

        # Generierung der Rekonstruktion
        outputs_flat = model(images_flat)

        # Rückformung der Daten in Bildformat (Batch, 1, 28, 28)
        outputs_img = outputs_flat.view(-1, 1, 28, 28)

        # Auswahl der ersten 8 Bilder für den Vergleich
        orig_imgs = images_original[:8].to(device)
        recon_imgs = outputs_img[:8]

        comparison = torch.cat([orig_imgs, recon_imgs])
        utils.save_image(comparison, "./ae_visuals/mnist_reconstruction.png", nrow=8)
        print("Vergleichsbild gespeichert: ./ae_visuals/mnist_reconstruction.png")

    # Extraktion der Latent-Space-Vektoren für den gesamten Datensatz
    print("Starte Extraktion der Features...")
    dataloader_full = DataLoader(dataset, batch_size=256, shuffle=False)
    latents, labels = [], []

    with torch.no_grad():
        for imgs, labs in dataloader_full:
            imgs = imgs.view(imgs.size(0), -1).to(device).float()
            z = model.encode(imgs)
            latents.append(z.cpu().numpy())
            labels.append(labs.numpy())

    # Zusammenfügen und Speichern der Arrays
    latents = np.vstack(latents)
    labels = np.hstack(labels)
    torch.save(model.state_dict(), "./ae_latents/mnist_autoencoder.pth")
    print("Modell-Gewichte gespeichert.")
    np.save("./ae_latents/mnist_linear_latents.npy", latents)
    np.save("./ae_latents/mnist_labels.npy", labels)
    print(f"Verarbeitung abgeschlossen. Datenform: {latents.shape}")