import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# ========== 1. KONFIGURATION & PFADE ======================
# ==========================================================

# Definition der Verzeichnisse
# XXX sowie Pfad muss natürlich angepasst werden, falls Sie das ausprobieren wollen
GAN_PATH = r"C:\Users\XXX\PyCharmMiscProject\generated\generated"
LABEL_PATH = r"C:\Users\XXX\PyCharmMiscProject\ae_latents"

# ==========================================================
# ========== 2. DATEN LADEN & VISUALISIEREN ================
# ==========================================================

# Laden der Bilddaten und Labels
images = np.load(os.path.join(GAN_PATH, "mnist_gan_images_10000.npy"))
labels = np.load(os.path.join(LABEL_PATH, "mnist_labels.npy"))

plt.figure(figsize=(15, 5))

# Iteration über die ersten 5 Samples
for i in range(5):
    plt.subplot(1, 5, i + 1)

    # Umformung des flachen Vektors in eine 2D-Matrix
    # Input-Dimension 1024 wird zu 32x32 Pixeln transformiert
    img_to_show = images[i].reshape(32, 32)

    plt.imshow(img_to_show, cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()