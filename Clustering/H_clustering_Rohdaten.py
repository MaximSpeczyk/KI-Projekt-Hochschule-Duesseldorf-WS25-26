import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix

# ==========================================================
# ========== KONFIGURATION =================================
# ==========================================================
# Definition der Parameter für die Analyse
# Hinweis: 30.000 Samples benötigen ca. 3.5 GB RAM für die Distanzmatrix
NUM_SAMPLES = 30000
CIFAR_PCA_COMPONENTS = 200


# ==========================================================
# ========== HILFSFUNKTIONEN ===============================
# ==========================================================

def plot_metrics_bars(k_vals, ari_list, nmi_list, sil_list, title, filename):
    """Erstellt ein Balkendiagramm zum Vergleich der Metriken bei verschiedenen k."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(k_vals))
    width = 0.25
    plt.bar(x - width, ari_list, width, label='ARI', color='skyblue')
    plt.bar(x, nmi_list, width, label='NMI', color='orange')
    plt.bar(x + width, sil_list, width, label='Silhouette', color='lightgreen')
    plt.xlabel('Anzahl der Cluster (k)')
    plt.ylabel('Score')
    plt.title(f'Hierarchical Metriken - {title} ({NUM_SAMPLES} Samples)')
    plt.xticks(x, k_vals)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_clustering_heatmap(y_true, y_pred, title, filename, k):
    """Visualisiert die Zuordnung von Clustern zu wahren Labels."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel(f'Gefundene Cluster ID (k={k})')
    plt.ylabel('Wahre Klasse (Label)')
    plt.title(f'Heatmap - {title} (Bester ARI bei k={k})')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def print_summary_table(dataset_name, k_values, ari_list, nmi_list, sil_list):
    """Gibt die Ergebnisse tabellarisch auf der Konsole aus."""
    print(f"\n--- Zusammenfassung Hierarchical: {dataset_name} ({NUM_SAMPLES} Samples) ---")
    print(f"{'k':>4} | {'ARI':>8} | {'NMI':>8} | {'Silhouette':>10}")
    print("-" * 45)
    for i in range(len(k_values)):
        print(f"{k_values[i]:>4} | {ari_list[i]:>8.4f} | {nmi_list[i]:>8.4f} | {sil_list[i]:>10.4f}")
    print("-" * 45)


# ==========================================================
# ========== 1. MNIST (Hierarchical) =======================
# ==========================================================
print(f"\n========== HIERARCHICAL – MNIST ({NUM_SAMPLES} Samples) ==========\n")

# Laden und Normalisieren der Daten
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Erstellung der Teilmenge (Subsampling)
X_mnist = mnist_train.data.numpy()[:NUM_SAMPLES].reshape(NUM_SAMPLES, -1)
y_mnist = mnist_train.targets.numpy()[:NUM_SAMPLES]

# Standardisierung und Dimensionsreduktion mittels PCA
X_mnist_scaled = StandardScaler().fit_transform(X_mnist)
X_mnist_pca = PCA(n_components=50).fit_transform(X_mnist_scaled)

k_vals = [8, 10, 12, 15]
mnist_ari, mnist_nmi, mnist_sil = [], [], []

best_ari_mnist = -1
best_labels_mnist = None
best_k_mnist = None

# Iteration über verschiedene Cluster-Anzahlen
for k in k_vals:
    print(f"  -> Berechne k={k} (Linkage='ward')...")
    # Initialisierung des Agglomerative Clustering
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hc.fit_predict(X_mnist_pca)

    current_ari = adjusted_rand_score(y_mnist, labels)
    mnist_ari.append(current_ari)
    mnist_nmi.append(normalized_mutual_info_score(y_mnist, labels))
    # Berechnung des Silhouette-Scores auf einer Teilmenge zur Performancesteigerung
    mnist_sil.append(silhouette_score(X_mnist_pca, labels, sample_size=10000))

    # Speicherung des Modells mit dem besten ARI-Wert
    if current_ari > best_ari_mnist:
        best_ari_mnist = current_ari
        best_labels_mnist = labels
        best_k_mnist = k

# Visualisierung der Ergebnisse
if best_labels_mnist is not None:
    plot_metrics_bars(k_vals, mnist_ari, mnist_nmi, mnist_sil, "MNIST", "mnist_hierarchical_metrics.png")
    plot_clustering_heatmap(y_mnist, best_labels_mnist, "MNIST", "mnist_hierarchical_heatmap.png", best_k_mnist)
print_summary_table("MNIST", k_vals, mnist_ari, mnist_nmi, mnist_sil)

# ==========================================================
# ========== 2. CIFAR-10 (Hierarchical) ====================
# ==========================================================
print(f"\n========== HIERARCHICAL – CIFAR-10 ({NUM_SAMPLES} Samples) ==========\n")

# Laden und Normalisieren der CIFAR-Daten
transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)

X_cifar = cifar_train.data[:NUM_SAMPLES].reshape(NUM_SAMPLES, -1)
y_cifar = np.array(cifar_train.targets)[:NUM_SAMPLES]

X_cifar_scaled = StandardScaler().fit_transform(X_cifar)

# Durchführung der PCA
print(f"Berechne PCA mit {CIFAR_PCA_COMPONENTS} Komponenten...")
X_cifar_pca = PCA(n_components=CIFAR_PCA_COMPONENTS).fit_transform(X_cifar_scaled)

cifar_ari, cifar_nmi, cifar_sil = [], [], []

best_ari_cifar = -1
best_labels_cifar = None
best_k_cifar = None

# Iteration über verschiedene Cluster-Anzahlen
for k in k_vals:
    print(f"  -> Berechne k={k} (Linkage='ward')...")
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hc.fit_predict(X_cifar_pca)

    current_ari = adjusted_rand_score(y_cifar, labels)
    cifar_ari.append(current_ari)
    cifar_nmi.append(normalized_mutual_info_score(y_cifar, labels))
    cifar_sil.append(silhouette_score(X_cifar_pca, labels, sample_size=10000))

    if current_ari > best_ari_cifar:
        best_ari_cifar = current_ari
        best_labels_cifar = labels
        best_k_cifar = k

# Visualisierung der Ergebnisse
if best_labels_cifar is not None:
    plot_metrics_bars(k_vals, cifar_ari, cifar_nmi, cifar_sil, "CIFAR-10", "cifar_hierarchical_metrics.png")
    plot_clustering_heatmap(y_cifar, best_labels_cifar, "CIFAR-10", "cifar_hierarchical_heatmap.png", best_k_cifar)
print_summary_table("CIFAR-10", k_vals, cifar_ari, cifar_nmi, cifar_sil)

print("\nVerarbeitung abgeschlossen. Clustering beendet.")