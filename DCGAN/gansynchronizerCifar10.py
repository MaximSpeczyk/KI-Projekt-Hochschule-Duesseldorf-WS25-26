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
MODEL_NAME = "discriminator_cifar10.pth"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

# Automatische Wahl der Hardware (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# ========== 2. ARCHITEKTUR (CIFAR-10) =====================
# ==========================================================

class CIFARDiscriminatorExtractor(nn.Module):
    """
    Modifizierter Diskriminator für CIFAR-10 zur Feature-Extraktion.
    Gibt die Ausgaben der letzten Convolutional-Layer zurück (Feature Matching).
    """

    def __init__(self, nc=3, ndf=64):
        super(CIFARDiscriminatorExtractor, self).__init__()
        # Definition der Schichten bis zur Feature-Ebene
        self.main = nn.Sequential(
            # 1. Layer: Input (3) x 32 x 32 -> (64) x 16 x 16
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 2. Layer: (64) x 16 x 16 -> (128) x 8 x 8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 3. Layer: (128) x 8 x 8 -> (256) x 4 x 4
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
    Lädt das CIFAR-10 Modell, extrahiert Features aus dem Datensatz
    und speichert diese als NumPy-Array.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Fehler: Modelldatei nicht gefunden: {MODEL_PATH}")
        return

    print(f"Starte Feature-Extraktion für CIFAR-10 auf: {DEVICE}")

    # Definition der Transformationen (Normalisierung auf [-1, 1] für RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Laden des Datensatzes und Erstellung eines Subsets
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Begrenzung auf 10.000 Samples für konsistente Analyse
    loader = DataLoader(Subset(dataset, range(10000)), batch_size=100, shuffle=False)

    # Initialisierung des Modells und Laden der Gewichte
    model = CIFARDiscriminatorExtractor().to(DEVICE)
    # 'strict=False' erlaubt das Laden, auch wenn der finale Layer fehlt (da wir Features wollen)
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
    save_file = os.path.join(MODEL_FOLDER, "cifar10_gan_features_clean.npy")
    np.save(save_file, X_clean)

    print(f"Extraktion abgeschlossen. Daten gespeichert unter: {save_file}")


if __name__ == "__main__":
    run_extraction()