import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
from sklearn.manifold import TSNE


# Hilfsfunktionen für die grafische Auswertung

def plot_metrics_bars(k_values, ari_list, nmi_list, sil_list, title, filename):
    """Erstellt ein Balkendiagramm zum Vergleich von ARI, NMI und Silhouette-Score."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(k_values))
    width = 0.25

    plt.bar(x - width, ari_list, width, label='ARI', color='skyblue')
    plt.bar(x, nmi_list, width, label='NMI', color='orange')
    plt.bar(x + width, sil_list, width, label='Silhouette', color='lightgreen')

    plt.xlabel('Anzahl der Cluster (k)')
    plt.ylabel('Score')
    plt.title(f'Clustering Metriken - {title}')
    plt.xticks(x, k_values)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.show()


def plot_clustering_heatmap(y_true, y_pred, title, filename):
    """
    Visualisiert die Verteilung der wahren Labels auf die gefundenen Cluster.
    Zeigt, welche Klassen welchem Cluster zugeordnet wurden.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Gefundene Cluster ID')
    plt.ylabel('Wahre Klasse (Label)')
    plt.title(f'Heatmap der Cluster-Zuweisung - {title}')
    plt.savefig(filename)
    plt.show()


# ==========================================================
# ========== MNIST ANALYSE =================================
# ==========================================================
print("\n========== KMEANS – MNIST (60.000 Samples) ==========\n")

# Laden und Normalisieren der Bilddaten
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Umformung der Daten (Flattening) für Scikit-Learn
X = mnist_train.data.numpy().reshape(len(mnist_train), -1)
y = mnist_train.targets.numpy()

# Vorverarbeitung: Skalierung und Dimensionsreduktion mittels PCA
X_scaled = StandardScaler().fit_transform(X)
X_reduced = PCA(n_components=50).fit_transform(X_scaled)

k_vals = [8, 10, 12]
mnist_ari, mnist_nmi, mnist_sil = [], [], []
best_labels_mnist = None

# Iteration über verschiedene Cluster-Anzahlen
for k in k_vals:
    print(f"Berechne MNIST für k={k}...")
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = km.fit_predict(X_reduced)

    # Berechnung der Qualitätsmetriken
    mnist_ari.append(adjusted_rand_score(y, labels))
    mnist_nmi.append(normalized_mutual_info_score(y, labels))
    mnist_sil.append(silhouette_score(X_reduced, labels))

    if k == 10:
        best_labels_mnist = labels

# Speichern und Anzeigen der Diagramme für MNIST
plot_metrics_bars(k_vals, mnist_ari, mnist_nmi, mnist_sil, "MNIST", "mnist_metrics.png")
plot_clustering_heatmap(y, best_labels_mnist, "MNIST (k=10)", "mnist_heatmap.png")

# ==========================================================
# ========== CIFAR-10 ANALYSE ==============================
# ==========================================================
print("\n========== KMEANS – CIFAR10 (50.000 Samples) ==========\n")

# Laden und Normalisieren des CIFAR-Datensatzes
transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)

X_cifar = cifar_train.data.reshape(len(cifar_train), -1)
y_cifar = np.array(cifar_train.targets)

# Vorverarbeitung: Skalierung und PCA (200 Komponenten für höhere Komplexität)
X_cifar_scaled = StandardScaler().fit_transform(X_cifar)
X_cifar_reduced = PCA(n_components=200).fit_transform(X_cifar_scaled)

cifar_ari, cifar_nmi, cifar_sil = [], [], []
best_labels_cifar = None

# Iteration über k-Werte für CIFAR-10
for k in k_vals:
    print(f"Berechne CIFAR10 für k={k}...")
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = km.fit_predict(X_cifar_reduced)

    cifar_ari.append(adjusted_rand_score(y_cifar, labels))
    cifar_nmi.append(normalized_mutual_info_score(y_cifar, labels))
    cifar_sil.append(silhouette_score(X_cifar_reduced, labels))

    if k == 10:
        best_labels_cifar = labels

# Speichern und Anzeigen der Diagramme für CIFAR-10
plot_metrics_bars(k_vals, cifar_ari, cifar_nmi, cifar_sil, "CIFAR-10", "cifar_metrics.png")
plot_clustering_heatmap(y_cifar, best_labels_cifar, "CIFAR-10 (k=10)", "cifar_heatmap.png")

# Berechnung der t-SNE Einbettung für eine Teilmenge (Visualisierung)
print("\nErstelle t-SNE für MNIST (Subset)...")
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X_reduced[:2000])

plt.figure(figsize=(10, 7))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=best_labels_mnist[:2000], cmap='tab10', alpha=0.7)
plt.title("t-SNE Visualisierung der Cluster (MNIST)")
plt.colorbar(label="Cluster ID")
plt.savefig("mnist_tsne.png")
plt.show()

# ==========================================================
# ========== ZUSAMMENFASSUNG DER ERGEBNISSE ================
# ==========================================================

def print_summary_table(dataset_name, k_values, ari_list, nmi_list, sil_list):
    """Gibt die Metriken in Tabellenform auf der Konsole aus."""
    print(f"\n--- Zusammenfassung Ergebnisse: {dataset_name} ---")
    print(f"{'k':>4} | {'ARI':>8} | {'NMI':>8} | {'Silhouette':>10}")
    print("-" * 45)
    for i in range(len(k_values)):
        print(f"{k_values[i]:>4} | {ari_list[i]:>8.4f} | {nmi_list[i]:>8.4f} | {sil_list[i]:>10.4f}")
    print("-" * 45)

# Ausgabe der finalen Tabellen
print_summary_table("MNIST", k_vals, mnist_ari, mnist_nmi, mnist_sil)
print_summary_table("CIFAR-10", k_vals, cifar_ari, cifar_nmi, cifar_sil)