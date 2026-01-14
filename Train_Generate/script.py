import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def simple_kmeans_example():
    """
    Ein einfaches K-Means-Clustering-Beispiel mit scikit-learn.
    """

    print("Starte K-Means-Beispiel...")

    # 1. Beispieldaten erstellen
    # Wir erstellen 300 Datenpunkte, die in 3 "Wolken" (centers=3)
    # in einem 2D-Raum (n_features=2) angeordnet sind.
    # X sind die Koordinaten (unsere Daten),
    # y_true sind die "echten" Labels (0, 1, oder 2), die wir aber K-Means nicht zeigen.
    X, y_true = make_blobs(n_samples=300,
                           centers=3,
                           n_features=2,
                           cluster_std=1.0,
                           random_state=42)

    print(f"Datenform (Samples, Features): {X.shape}")

    # 2. K-Means-Modell initialisieren
    # Wir sagen dem Algorithmus, dass er 3 Cluster (n_clusters=3) finden soll.
    # 'n_init=10' bedeutet, dass der Algorithmus 10 Mal mit verschiedenen
    # Startpunkten läuft und das beste Ergebnis zurückgibt.
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)

    # 3. Modell trainieren (Clustering durchführen)
    # Das ist der eigentliche Clustering-Schritt.
    kmeans.fit(X)

    # 4. Ergebnisse abrufen
    # y_kmeans sind die Cluster-Zuweisungen (Labels), die K-Means gefunden hat.
    y_kmeans = kmeans.labels_

    # centers sind die Koordinaten der gefundenen Cluster-Zentren.
    centers = kmeans.cluster_centers_

    print("\nClustering abgeschlossen.")
    print("Gefundene Cluster-Zentren (Koordinaten):")
    print(centers)

    # 5. Ergebnisse visualisieren (optional, aber sehr hilfreich)
    # Speichert die Grafik als PNG-Datei
    plt.figure(figsize=(8, 6))

    # Plotte die Datenpunkte, eingefärbt nach dem gefundenen Cluster (y_kmeans)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.7, label='Datenpunkte')

    # Plotte die gefundenen Cluster-Zentren
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Cluster-Zentren')

    plt.title('K-Means Clustering Ergebnis')
    plt.xlabel('Merkmal 1')
    plt.ylabel('Merkmal 2')
    plt.legend()
    plt.grid(True)

    # Speichere die Plot-Datei
    plot_filename = 'kmeans_clustering.png'
    plt.savefig(plot_filename)

    print(f"\nVisualisierung wurde als '{plot_filename}' gespeichert.")


# Das Skript ausführen
if __name__ == "__main__":
    simple_kmeans_example()