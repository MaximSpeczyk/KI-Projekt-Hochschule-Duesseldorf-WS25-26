import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix


# ==========================================================
# ========== HILFSFUNKTIONEN ===============================
# ==========================================================

def plot_metrics_bars(params, ari_list, nmi_list, sil_list, title, filename):
    """Erstellt ein Balkendiagramm für die Metriken ARI, NMI und Silhouette."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(params))
    width = 0.25
    plt.bar(x - width, ari_list, width, label='ARI', color='skyblue')
    plt.bar(x, nmi_list, width, label='NMI', color='orange')
    plt.bar(x + width, sil_list, width, label='Silhouette', color='lightgreen')
    plt.xlabel('Parameter epsilon (eps)')
    plt.ylabel('Score')
    plt.title(f'DBSCAN Metriken - {title} (FULL DATASET)')
    plt.xticks(x, params)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_clustering_heatmap(y_true, y_pred, title, filename, used_eps):
    """Visualisiert die Zuordnung von Datenpunkten zu Clustern (inkl. Noise)."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Gefundene Cluster ID (-1 = Noise)')
    plt.ylabel('Wahre Klasse (Label)')
    plt.title(f'Heatmap - {title} (eps={used_eps})')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def print_summary_table(dataset_name, eps_values, ari_list, nmi_list, sil_list, clusters_found):
    """Gibt eine Zusammenfassung der Ergebnisse pro Epsilon-Wert aus."""
    print(f"\n--- Zusammenfassung DBSCAN: {dataset_name} (FULL) ---")
    print(f"{'eps':>5} | {'Clusters':>8} | {'ARI':>8} | {'NMI':>8} | {'Sil':>8}")
    print("-" * 55)
    for i in range(len(eps_values)):
        sil_str = f"{sil_list[i]:>8.4f}" if sil_list[i] is not None else "     nan"
        print(f"{eps_values[i]:>5} | {clusters_found[i]:>8} | {ari_list[i]:>8.4f} | {nmi_list[i]:>8.4f} | {sil_str}")
    print("-" * 55)


# ==========================================================
# ========== 1. MNIST DBSCAN (Vollständiger Datensatz) =====
# ==========================================================
print(f"\n========== DBSCAN – MNIST (60.000 Samples) ==========\n")

# Laden und Normalisieren der Daten
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Nutzung des gesamten Datensatzes ohne Subsampling
X_mnist = mnist_train.data.numpy().reshape(-1, 28 * 28)
y_mnist = mnist_train.targets.numpy()

# Standardisierung und PCA-Reduktion auf 50 Dimensionen
X_mnist_scaled = StandardScaler().fit_transform(X_mnist)
X_mnist_pca = PCA(n_components=50).fit_transform(X_mnist_scaled)

# Definition der Epsilon-Werte für die Dichtebestimmung
eps_vals_mnist = [4.0, 5.0, 6.0, 7.0, 8.0]
mnist_ari, mnist_nmi, mnist_sil, mnist_clusters = [], [], [], []

best_ari_mnist = -1
best_labels_mnist = None
best_eps_mnist = None

# Iteration über verschiedene Epsilon-Werte
for e in eps_vals_mnist:
    print(f"  -> Berechne DBSCAN mit eps={e} ...")
    # min_samples angepasst für Robustheit bei 60k Datenpunkten
    db = DBSCAN(eps=e, min_samples=15, n_jobs=-1)
    labels = db.fit_predict(X_mnist_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    current_ari = adjusted_rand_score(y_mnist, labels)
    current_nmi = normalized_mutual_info_score(y_mnist, labels)

    # Schätzung des Silhouette-Scores (Sample 10k) zur Laufzeitoptimierung
    if n_clusters > 1 and len(set(labels)) > 1:
        current_sil = silhouette_score(X_mnist_pca, labels, sample_size=10000)
    else:
        current_sil = None

    mnist_clusters.append(n_clusters)
    mnist_ari.append(current_ari)
    mnist_nmi.append(current_nmi)
    mnist_sil.append(current_sil)

    if current_ari > best_ari_mnist:
        best_ari_mnist = current_ari
        best_labels_mnist = labels
        best_eps_mnist = e

# Speichern der Ergebnisse für MNIST
if best_labels_mnist is not None:
    plot_metrics_bars(eps_vals_mnist, mnist_ari, mnist_nmi, [s if s else 0 for s in mnist_sil], "MNIST",
                      "generated/mnist_dbscan_metrics.png")
    plot_clustering_heatmap(y_mnist, best_labels_mnist, "MNIST", "generated/mnist_dbscan_heatmap.png", best_eps_mnist)
print_summary_table("MNIST", eps_vals_mnist, mnist_ari, mnist_nmi, mnist_sil, mnist_clusters)

# ==========================================================
# ========== 2. CIFAR-10 DBSCAN (Vollständiger Datensatz) ==
# ==========================================================
print(f"\n========== DBSCAN – CIFAR-10 (50.000 Samples) ==========\n")

# Laden und Transformation des CIFAR-Datensatzes
transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)

X_cifar = cifar_train.data.reshape(-1, 32 * 32 * 3)
y_cifar = np.array(cifar_train.targets)

X_cifar_scaled = StandardScaler().fit_transform(X_cifar)
X_cifar_pca = PCA(n_components=200).fit_transform(X_cifar_scaled)

# Höhere Epsilon-Werte für CIFAR aufgrund größerer Varianz in den Daten
eps_vals_cifar = [25.0, 30.0, 35.0, 40.0, 45.0]
cifar_ari, cifar_nmi, cifar_sil, cifar_clusters = [], [], [], []

best_ari_cifar = -1
best_labels_cifar = None
best_eps_cifar = None

# Iteration über Epsilon-Werte für CIFAR-10
for e in eps_vals_cifar:
    print(f"  -> Berechne DBSCAN mit eps={e} ...")
    db = DBSCAN(eps=e, min_samples=15, n_jobs=-1)
    labels = db.fit_predict(X_cifar_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    current_ari = adjusted_rand_score(y_cifar, labels)
    current_nmi = normalized_mutual_info_score(y_cifar, labels)

    if n_clusters > 1 and len(set(labels)) > 1:
        current_sil = silhouette_score(X_cifar_pca, labels, sample_size=10000)
    else:
        current_sil = None

    cifar_clusters.append(n_clusters)
    cifar_ari.append(current_ari)
    cifar_nmi.append(current_nmi)
    cifar_sil.append(current_sil)

    if current_ari > best_ari_cifar:
        best_ari_cifar = current_ari
        best_labels_cifar = labels
        best_eps_cifar = e

# Speichern der Ergebnisse für CIFAR-10
if best_labels_cifar is not None:
    plot_metrics_bars(eps_vals_cifar, cifar_ari, cifar_nmi, [s if s else 0 for s in cifar_sil], "CIFAR-10",
                      "generated/cifar_dbscan_metrics.png")
    plot_clustering_heatmap(y_cifar, best_labels_cifar, "CIFAR-10", "generated/cifar_dbscan_heatmap.png", best_eps_cifar)
print_summary_table("CIFAR-10", eps_vals_cifar, cifar_ari, cifar_nmi, cifar_sil, cifar_clusters)

print("\nVerarbeitung des gesamten Datensatzes abgeschlossen.")