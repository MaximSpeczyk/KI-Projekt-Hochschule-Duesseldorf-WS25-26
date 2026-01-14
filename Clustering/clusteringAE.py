import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix

# ==========================================================
# ========== AE-CLUSTERING ANALYSE SCRIPT ==================
# ==========================================================

# Erstellung des Verzeichnisses für die Ergebnisse
os.makedirs("./ae_clustering_results", exist_ok=True)


# --- Hilfsfunktionen für die Visualisierung ---

def plot_metrics_bars(params, ari_list, nmi_list, sil_list, title, filename, xlabel='k'):
    """Erstellt ein Balkendiagramm für ARI, NMI und Silhouette-Score."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(params))
    width = 0.25
    plt.bar(x - width, ari_list, width, label='ARI', color='skyblue')
    plt.bar(x, nmi_list, width, label='NMI', color='orange')
    plt.bar(x + width, sil_list, width, label='Silhouette', color='lightgreen')
    plt.xlabel(xlabel)
    plt.ylabel('Score')
    plt.title(f'AE-Clustering Metriken - {title}')
    plt.xticks(x, params)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_clustering_heatmap(y_true, y_pred, title, filename):
    """Generiert eine Heatmap der Konfusionsmatrix zwischen Labels und Clustern."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Gefundene Cluster ID')
    plt.ylabel('Wahre Klasse (Label)')
    plt.title(f'Heatmap - {title}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def print_summary(name, params, ari, nmi, sil, param_name='k'):
    """Gibt eine tabellarische Zusammenfassung der Metriken auf der Konsole aus."""
    print(f"\n--- Zusammenfassung AE-Clustering: {name} ---")
    print(f"{param_name:>5} | {'ARI':>8} | {'NMI':>8} | {'Sil':>8}")
    print("-" * 45)
    for i in range(len(params)):
        s_val = f"{sil[i]:.4f}" if sil[i] is not None else "   nan"
        print(f"{params[i]:>5} | {ari[i]:>8.4f} | {nmi[i]:>8.4f} | {s_val:>8}")


# --- Hauptfunktion der Analyse ---

def perform_ae_analysis(X_path, y_path, dataset_name):
    """
    Führt K-Means, Hierarchical Clustering und DBSCAN auf den Latent Features aus.
    Berechnet Metriken und speichert Diagramme.
    """
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Fehler: Dateien für {dataset_name} nicht gefunden ({X_path}).")
        return

    print(f"\n========== Starte Analyse: {dataset_name} (Latent Space) ==========")
    # Laden der Latent-Space-Daten und Labels
    X = np.load(X_path)
    y = np.load(y_path)

    # 1. K-Means Clustering
    print(f"[{dataset_name}] Berechne K-Means...")
    k_vals = [10, 12, 14, 16]
    ari_km, nmi_km, sil_km = [], [], []
    for k in k_vals:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        lbls = km.fit_predict(X)
        ari_km.append(adjusted_rand_score(y, lbls))
        nmi_km.append(normalized_mutual_info_score(y, lbls))
        sil_km.append(silhouette_score(X, lbls, sample_size=10000))

        # Erstellung der Heatmap für k=10
        if k == 10:
            plot_clustering_heatmap(y, lbls, f"{dataset_name} AE-KMeans (k=10)",
                                    f"./ae_clustering_results/{dataset_name}_kmeans_heatmap.png")

    # Speichern der Ergebnisse für K-Means
    plot_metrics_bars(k_vals, ari_km, nmi_km, sil_km, f"{dataset_name} AE-KMeans",
                      f"./ae_clustering_results/{dataset_name}_kmeans_metrics.png")
    print_summary(f"{dataset_name} KMeans", k_vals, ari_km, nmi_km, sil_km)

    # 2. Hierarchical Clustering (Agglomerative)
    print(f"[{dataset_name}] Berechne Hierarchical Clustering (Subsample 30k)...")
    # Subsampling für bessere Performance bei großen Datensätzen
    sub_size = 30000
    if len(X) > sub_size:
        idx = np.random.choice(len(X), sub_size, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    k_h = [6, 8, 10, 12, 14, 16]
    ari_h, nmi_h, sil_h = [], [], []
    best_ari, best_lbls, best_k = -1, None, None

    for k in k_h:
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        lbls = hc.fit_predict(X_sub)
        ari_v = adjusted_rand_score(y_sub, lbls)
        ari_h.append(ari_v)
        nmi_h.append(normalized_mutual_info_score(y_sub, lbls))
        sil_h.append(silhouette_score(X_sub, lbls, sample_size=10000))

        # Tracking des besten Ergebnisses
        if ari_v > best_ari:
            best_ari, best_lbls, best_k = ari_v, lbls, k

    plot_metrics_bars(k_h, ari_h, nmi_h, sil_h, f"{dataset_name} AE-Hierarchical",
                      f"./ae_clustering_results/{dataset_name}_hier_metrics.png")
    plot_clustering_heatmap(y_sub, best_lbls, f"{dataset_name} AE-Hierarchical (Best k={best_k})",
                            f"./ae_clustering_results/{dataset_name}_hier_heatmap.png")
    print_summary(f"{dataset_name} Hierarchical", k_h, ari_h, nmi_h, sil_h)

    # 3. DBSCAN Clustering
    print(f"[{dataset_name}] Berechne DBSCAN...")
    # Definition der Epsilon-Werte abhängig vom Datensatz
    if dataset_name == "MNIST":
        eps_vals = [0.5]
    else:
        eps_vals = [1.0]

    ari_d, nmi_d, sil_d = [], [], []
    best_ari_db, best_lbls_db, best_eps = -1, None, None

    for eps in eps_vals:
        db = DBSCAN(eps=eps, min_samples=15)
        lbls = db.fit_predict(X)
        n_cl = len(set(lbls)) - (1 if -1 in lbls else 0)
        ari_v = adjusted_rand_score(y, lbls)
        ari_d.append(ari_v)
        nmi_d.append(normalized_mutual_info_score(y, lbls))

        # Silhouette-Score nur berechenbar bei > 1 Cluster
        if n_cl > 1:
            sil_d.append(silhouette_score(X, lbls, sample_size=10000))
        else:
            sil_d.append(None)

        if ari_v > best_ari_db:
            best_ari_db, best_lbls_db, best_eps = ari_v, lbls, eps

    plot_metrics_bars(eps_vals, ari_d, nmi_d, [s if s else 0 for s in sil_d], f"{dataset_name} AE-DBSCAN",
                      f"./ae_clustering_results/{dataset_name}_dbscan_metrics.png", xlabel='eps')
    if best_lbls_db is not None:
        plot_clustering_heatmap(y, best_lbls_db, f"{dataset_name} AE-DBSCAN (Best eps={best_eps})",
                                f"./ae_clustering_results/{dataset_name}_dbscan_heatmap.png")
    print_summary(f"{dataset_name} DBSCAN", eps_vals, ari_d, nmi_d, sil_d, param_name='eps')


# --- Start der Skript-Ausführung ---

# Definition des Basis-Pfads für Eingabedaten
base_path = r"C:\Users\budak\PyCharmMiscProject\ae_latents"

# Ausführung der Analyse für MNIST
perform_ae_analysis(
    X_path=os.path.join(base_path, "mnist_linear_latents.npy"),
    y_path=os.path.join(base_path, "mnist_labels.npy"),
    dataset_name="MNIST"
)

# Ausführung der Analyse für CIFAR-10
perform_ae_analysis(
    X_path=os.path.join(base_path, "cifar_conv_latents.npy"),
    y_path=os.path.join(base_path, "cifar_labels.npy"),
    dataset_name="CIFAR-10"
)

print("\nAlle Analysen abgeschlossen. Ergebnisse in 'ae_clustering_results' gespeichert.")