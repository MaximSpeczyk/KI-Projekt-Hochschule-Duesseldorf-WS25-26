import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix

# ==========================================================
# ========== 1. SETUP & KONFIGURATION ======================
# ==========================================================

# Definition der Pfade zu den generierten Features und Labels
# XXX sowie Pfad muss natürlich angepasst werden, falls Sie das ausprobieren wollen
BASE_PATH_GAN = r"C:\Users\XXX\PyCharmMiscProject\generated\generated"
BASE_PATH_LABELS = r"C:\Users\XXX\PyCharmMiscProject\ae_latents"
MAIN_SAVE_DIR = "./hierarchical_final_results"

# Konfiguration der Parameter für Hierarchisches Clustering
datasets = {
    "MNIST": {
        "features": "mnist_gan_features_clean.npy",
        "labels": "mnist_labels.npy",
        "k_range": [8, 10, 12, 15, 20]
    },
    "CIFAR-10": {
        "features": "cifar10_gan_features_clean.npy",
        "labels": "cifar_labels.npy",
        "k_range": [8, 10, 12, 15, 20]
    }
}


# ==========================================================
# ========== 2. ANALYSE-LOGIK ==============================
# ==========================================================

def run_hierarchical_analysis(name, config):
    """
    Führt Agglomerative Clustering auf den Features aus und
    speichert die Ergebnisse (Metriken, Diagramme, Heatmaps).
    """
    print(f"\nStarte hierarchische Analyse für: {name}")
    save_path = os.path.join(MAIN_SAVE_DIR, name)
    os.makedirs(save_path, exist_ok=True)

    # Laden der Feature-Daten und der zugehörigen Labels
    X = np.load(os.path.join(BASE_PATH_GAN, config["features"]))
    y = np.load(os.path.join(BASE_PATH_LABELS, config["labels"]))[:X.shape[0]]

    results = []
    best_ari = -1
    best_labels = None
    best_k = None

    print(f"{'K-Wert':<10} | {'ARI':<8} | {'NMI':<8} | {'Silh':<8}")
    print("-" * 50)

    # Iteration über die definierten Cluster-Anzahlen (k)
    for k in config["k_range"]:
        # Durchführung des hierarchischen Clusterings (Ward-Linkage)
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = hc.fit_predict(X)

        # Berechnung der Clustering-Metriken
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        # Berechnung des Silhouette-Scores (Stichprobe zur Performance-Optimierung)
        sil = silhouette_score(X, labels, sample_size=2000)

        print(f"{k:<10} | {ari:.4f} | {nmi:.4f} | {sil:.4f}")

        results.append({"K": k, "ARI": ari, "NMI": nmi, "Silhouette": sil})

        # Speicherung des besten Ergebnisses (basierend auf ARI)
        if ari > best_ari:
            best_ari = ari
            best_labels = labels
            best_k = k

    # --- Visualisierung der Ergebnisse ---

    # 1. Erstellung des Balkendiagramms für den Metrik-Vergleich
    df = pd.DataFrame(results)
    df_melted = df.melt(id_vars="K", var_name="Metrik", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="K", y="Score", hue="Metrik")
    plt.title(f'Hierarchisches Clustering Metrik-Vergleich: {name}')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_path, f"metrics_comparison_{name}.png"))
    plt.close()

    # 2. Speicherung der Heatmap für das beste Ergebnis (Best K)
    if best_labels is not None:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y, best_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'Beste Heatmap {name} (K={best_k})\nARI: {best_ari:.4f}')
        plt.xlabel("Vorhergesagtes Cluster")
        plt.ylabel("Wahre Klasse")
        plt.savefig(os.path.join(save_path, f"BEST_heatmap_K{best_k}.png"))
        plt.close()
        print(f"Bestes K für {name}: {best_k} (ARI: {best_ari:.4f})")


# ==========================================================
# ========== 3. AUSFÜHRUNG =================================
# ==========================================================

# Erstellung des Ausgabeverzeichnisses
os.makedirs(MAIN_SAVE_DIR, exist_ok=True)

# Iteration über alle konfigurierten Datensätze
for ds_name, ds_config in datasets.items():
    try:
        run_hierarchical_analysis(ds_name, ds_config)
    except Exception as e:
        print(f"Fehler bei {ds_name}: {e}")

print(f"\nAnalyse abgeschlossen. Ergebnisse gespeichert unter: {os.path.abspath(MAIN_SAVE_DIR)}")