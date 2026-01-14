import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# ==========================================================
# ========== 1. KONFIGURATION & PFADE ======================
# ==========================================================

# Definition der Pfade zum trainierten Modell und Speicherort
# XXX sowie Pfad muss natürlich angepasst werden, falls Sie das ausprobieren wollen
MODEL_FOLDER = r"C:\Users\XXX\PyCharmMiscProject\generated\generated"
MODEL_NAME = "discriminator_mnist.pth"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

# Automatische Wahl der Hardware (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# ========== 2. ARCHITEKTUR (MNIST) ========================
# ==========================================================

class DiscriminatorFeatureExtractor(nn.Module):
    """
    Modifizierter Diskriminator für MNIST zur Feature-Extraktion.
    Die Architektur entspricht den ersten 8 Schichten des trainierten Modells.
    """

    def __init__(self, nc=1, ndf=64):
        super(DiscriminatorFeatureExtractor, self).__init__()
        # Definition der Schichten bis zur Feature-Ebene
        self.main = nn.Sequential(
            # 1. Layer: Input (nc) x 32 x 32 -> (ndf) x 16 x 16
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 2. Layer: (ndf) x 16 x 16 -> (ndf*2) x 8 x 8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 3. Layer: (ndf*2) x 8 x 8 -> (ndf*4) x 4 x 4
            # Dies ist die Ebene, auf der die Features extrahiert werden
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        """Durchlauf durch die Schichten und Flattening der Ausgabe."""
        features = self.main(x)
        return torch.flatten(features, 1)  # Transformation in flachen Feature-Vektor


# ==========================================================
# ========== 3. FEATURE-EXTRAKTION =========================
# ==========================================================

def run_extraction():
    """
    Lädt das MNIST Modell, extrahiert Features aus dem Datensatz
    und speichert diese als NumPy-Array.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Fehler: Modelldatei nicht gefunden: {MODEL_PATH}")
        return

    print(f"Starte Feature-Extraktion für MNIST auf: {DEVICE}")

    # Definition der Transformationen
    # Resize auf 32x32 ist notwendig für die Konsistenz der Architektur
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Laden des Datensatzes und Erstellung eines Subsets
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 'shuffle=False' stellt sicher, dass die Reihenfolge den Labels entspricht
    loader = DataLoader(Subset(dataset, range(10000)), batch_size=100, shuffle=False)

    # Initialisierung des Modells
    model = DiscriminatorFeatureExtractor().to(DEVICE)

    # Laden der Gewichte mit 'strict=False', da der finale Layer entfernt wurde
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_features = []

    # Extraktionsschleife ohne Gradientenberechnung
    with torch.no_grad():
        for imgs, _ in loader:
            feat = model(imgs.to(DEVICE))
            all_features.append(feat.cpu().numpy())

    # Zusammenfügen und Speichern der Daten
    X_clean = np.concatenate(all_features)
    save_file = os.path.join(MODEL_FOLDER, "mnist_gan_features_clean.npy")
    np.save(save_file, X_clean)

    print(f"Extraktion abgeschlossen. Daten gespeichert unter: {save_file}")


if __name__ == "__main__":
    run_extraction()