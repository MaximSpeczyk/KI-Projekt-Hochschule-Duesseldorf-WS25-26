import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix

# ==========================================================
# ========== 1. SETUP & KONFIGURATION ======================
# ==========================================================

# Definition der Pfade zu den generierten Features und Labels
BASE_PATH_GAN = r"C:\Users\budak\PyCharmMiscProject\generated\generated"
BASE_PATH_LABELS = r"C:\Users\budak\PyCharmMiscProject\ae_latents"
MAIN_SAVE_DIR = "./dbscan_final_results"

datasets = {
    "MNIST": {
        "features": "mnist_gan_features_clean.npy",
        "labels": "mnist_labels.npy",
        "eps_range": [26, 28, 30, 32, 34],
        "min_samples": 15  # Parameterwahl gemäß Literatur [185]
    },
    "CIFAR-10": {
        "features": "cifar10_gan_features_clean.npy",
        "labels": "cifar_labels.npy",
        # Anpassung des Epsilon-Wertes für CIFAR-10 aufgrund höherer Komplexität
        "eps_range": [32.0, 34.0, 36.0, 38.0, 40.0],
        "min_samples": 15  # Parameterwahl gemäß Literatur [185]
    }
}


# ==========================================================
# ========== 2. ANALYSE-LOGIK ==============================
# ==========================================================

def run_dbscan_analysis(name, config):
    """
    Führt DBSCAN auf den geladenen Features aus und speichert
    die Ergebnisse (Metriken, Heatmaps, Diagramme).
    """
    print(f"\nStarte Analyse für: {name}")
    save_path = os.path.join(MAIN_SAVE_DIR, name)
    os.makedirs(save_path, exist_ok=True)

    # Laden der Feature-Daten und der zugehörigen Labels
    X = np.load(os.path.join(BASE_PATH_GAN, config["features"]))
    y = np.load(os.path.join(BASE_PATH_LABELS, config["labels"]))[:X.shape[0]]

    results = []
    print(f"{'Epsilon':<10} | {'ARI':<8} | {'NMI':<8} | {'Silh':<8} | {'Clusters'} | {'Noise %'}")
    print("-" * 70)

    # Iteration über die definierten Epsilon-Werte
    for eps in config["eps_range"]:
        db = DBSCAN(eps=eps, min_samples=config["min_samples"])
        labels = db.fit_predict(X)

        # Berechnung der Clustering-Metriken (ARI, NMI)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        # Berechnung des Silhouette-Scores (nur falls Cluster vorhanden sind)
        sil = silhouette_score(X, labels, sample_size=2000) if num_clusters > 1 else 0.0
        noise_perc = (labels == -1).sum() / len(labels) * 100

        print(f"{eps:<10.1f} | {ari:.4f} | {nmi:.4f} | {sil:.4f} | {num_clusters:<8} | {noise_perc:.1f}%")

        results.append({"Epsilon": eps, "ARI": ari, "NMI": nmi, "Silhouette": sil})

        # Erstellung und Speicherung der Konfusionsmatrix als Heatmap
        if num_clusters > 0:
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix(y, labels), annot=True, fmt='d', cmap='YlGnBu')
            plt.title(f'{name} DBSCAN (eps={eps}) - ARI: {ari:.4f}')
            plt.savefig(os.path.join(save_path, f"heatmap_eps_{eps}.png"))
            plt.close()

    # Visualisierung der Ergebnisse als Balkendiagramm
    df = pd.DataFrame(results)
    df.plot(x="Epsilon", y=["ARI", "NMI", "Silhouette"], kind="bar", figsize=(10, 6))
    plt.title(f'Metrik-Vergleich {name} (DBSCAN)')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_path, f"metrics_comparison_{name}.png"))
    plt.close()


# ==========================================================
# ========== 3. AUSFÜHRUNG =================================
# ==========================================================

for ds_name, ds_config in datasets.items():
    try:
        run_dbscan_analysis(ds_name, ds_config)
    except Exception as e:
        print(f"Fehler bei {ds_name}: {e}")

print(f"\nAnalyse abgeschlossen. Ergebnisse gespeichert in: {os.path.abspath(MAIN_SAVE_DIR)}")