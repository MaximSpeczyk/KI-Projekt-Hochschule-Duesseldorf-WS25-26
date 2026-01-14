import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix

# Konfiguration und Pfad-Definitionen
# XXX muss natürlich angepasst werden, falls Sie das ausprobieren wollen
BASE_PATH = r"C:\Users\XXX\PyCharmMiscProject\ae_latents"
os.makedirs("./dbscan_ae_results", exist_ok=True)

# Definition der zu testenden Epsilon-Bereiche
EPS_CONFIG = {
    "MNIST": [5.0],
    "CIFAR-10": [3.0, 4.0, 5.0, 6.0]
}


# Hilfsfunktionen für Plotting und Analyse

def plot_dbscan_metrics(eps_vals, ari_list, nmi_list, sil_list, title, filename):
    """Visualisiert ARI, NMI und Silhouette-Score als Balkendiagramm."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(eps_vals))
    width = 0.25

    plt.bar(x - width, ari_list, width, label='ARI', color='skyblue')
    plt.bar(x, nmi_list, width, label='NMI', color='orange')
    plt.bar(x + width, sil_list, width, label='Silhouette', color='lightgreen')

    plt.xlabel('Parameter epsilon (eps)')
    plt.ylabel('Score')
    plt.title(f'AE-DBSCAN Metriken - {title}')
    plt.xticks(x, eps_vals)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_dbscan_heatmap(y_true, y_pred, title, filename, eps):
    """Erstellt und speichert eine Heatmap der Konfusionsmatrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Cluster ID (-1 = Noise)')
    plt.ylabel('Wahre Klasse (Label)')
    plt.title(f'Heatmap - {title} (AE-Latents, eps={eps})')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def run_dbscan_search(dataset_name, x_file, y_file):
    """
    Lädt Daten, führt DBSCAN mit verschiedenen Epsilon-Werten aus
    und speichert die Metriken sowie die beste Heatmap.
    """
    x_path = os.path.join(BASE_PATH, x_file)
    y_path = os.path.join(BASE_PATH, y_file)

    if not os.path.exists(x_path):
        print(f"Fehler: Datei nicht gefunden: {x_path}")
        return

    # Laden der Datensätze (Latents und Labels)
    print(f"\nLade Daten für {dataset_name}...")
    X = np.load(x_path)
    y = np.load(y_path)

    eps_list = EPS_CONFIG[dataset_name]
    ari_results, nmi_results, sil_results = [], [], []

    print(f"{'eps':>5} | {'Cluster':>8} | {'Noise %':>8} | {'ARI':>8} | {'NMI':>8} | {'Sil':>8}")
    print("-" * 65)

    best_ari = -1
    best_labels = None
    best_eps = None

    # Iteration über Hyperparameter
    for eps in eps_list:
        # Initialisierung und Fitting von DBSCAN
        db = DBSCAN(eps=eps, min_samples=15, n_jobs=-1)
        labels = db.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_perc = (labels == -1).sum() / len(labels) * 100

        # Berechnung der Clustering-Metriken
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        sil = 0
        if n_clusters > 1:
            sil = silhouette_score(X, labels, sample_size=10000)

        ari_results.append(ari)
        nmi_results.append(nmi)
        sil_results.append(sil)

        print(f"{eps:>5.1f} | {n_clusters:>8} | {noise_perc:>7.1f}% | {ari:>8.4f} | {nmi:>8.4f} | {sil:>8.4f}")

        # Tracking des besten Ergebnisses (basierend auf ARI)
        if ari > best_ari:
            best_ari = ari
            best_labels = labels
            best_eps = eps

    # Speichern der Ergebnisse
    metrics_fname = f"./dbscan_ae_results/{dataset_name.lower()}_dbscan_metrics.png"
    plot_dbscan_metrics(eps_list, ari_results, nmi_results, sil_results, dataset_name, metrics_fname)
    print(f"Balkendiagramm gespeichert unter: {metrics_fname}")

    if best_labels is not None:
        heatmap_fname = f"./dbscan_ae_results/{dataset_name.lower()}_best_dbscan_heatmap.png"
        plot_dbscan_heatmap(y, best_labels, dataset_name, heatmap_fname, best_eps)
        print(f"Beste Heatmap (eps={best_eps}) gespeichert unter: {heatmap_fname}")


# Start der Analyse-Logik
if __name__ == "__main__":
    # Ausführung für MNIST
    run_dbscan_search("MNIST", "mnist_linear_latents.npy", "mnist_labels.npy")

    # Ausführung für CIFAR-10
    run_dbscan_search("CIFAR-10", "cifar_conv_latents.npy", "cifar_labels.npy")

    print("\nVerarbeitung abgeschlossen.")