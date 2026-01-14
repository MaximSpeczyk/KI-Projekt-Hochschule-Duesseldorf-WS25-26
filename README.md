# 🕵️‍♂️ Clustering: MNIST vs. CIFAR-10

Willkommen in unserem Repo! Dieses Projekt entstand im Rahmen unseres Mastermoduls "Künstliche Intelligenz" (PO2022).

**Die Kernfrage:** Können wir Computer dazu bringen, Bilder sinnvoll zu gruppieren, ohne ihnen vorher zu sagen, was auf den Bildern zu sehen ist? (Unsupervised Learning)

## 💡 Worum geht's?

Wir haben uns angeschaut, wie gut klassische Clustering-Methoden mit Bilddaten klarkommen. Dabei haben wir schnell gemerkt: Einfach nur Pixel vergleichen ("Rohdaten") funktioniert bei komplexen Bildern nicht wirklich gut.

Deshalb haben wir **Deep Learning** zur Hilfe geholt. Wir nutzen Autoencoder und GANs, um die Bilder erst zu "verstehen" und dann zu sortieren.

**Unsere Datensätze:**
* 🟢 **MNIST:** Handgeschriebene Ziffern 
* 🔴 **CIFAR-10:** Echte Fotos von Autos, Tieren, etc.

## 🛠️ Was wir gebaut haben

Wir haben das Ganze in **Python** gebaut. Hier sind unsere wichtigsten Werkzeuge:

* 🧠 **PyTorch & Torchvision:** Für die neuronalen Netze (Convolutional Autoencoder & DCGAN).
* 🧮 **Scikit-learn:** Für die Clustering-Algorithmen (K-Means, DBSCAN, Hierarchisch) und Metriken.
* 📊 **Matplotlib & Seaborn:** Damit die Ergebnisse auch gut aussehen.
* 🐼 **Pandas & NumPy:** Für das Daten-Management.

## 🧪 Die Experimente

Wir haben drei Szenarien durchgespielt:
1.  **Baseline:** Clustering direkt auf den Pixeln (mit PCA reduziert).
2.  **Autoencoder:** Clustering auf dem komprimierten "Wissen" (Latent Features) eines trainierten Autoencoders.
3.  **GANs:** Versuch, synthetische Daten zur Verbesserung zu nutzen.

## 📉 Die Ergebnisse

*(Detaillierte Ergebnisse findet ihr in der Dokumentation im Ordner `/docs`)*

* **MNIST:** Hier klappt fast alles super. Selbst einfache Methoden können die Ziffern gut trennen.
* **CIFAR-10:** Das war eine harte Nuss. Auf Rohdaten versagen die Algorithmen fast komplett (alles ist ein großer Brei). Mit dem **Convolutional Autoencoder** konnten wir die Ergebnisse deutlich verbessern, aber es bleibt eine Herausforderung.
* **DBSCAN:** Hatte große Probleme mit der unterschiedlichen Dichte der Daten (Entweder alles ist Rauschen oder alles ist ein Cluster).

## Selbst ausprobieren
Wir haben ein kleines GUI programmiert, so dass der Code einfach zu benutzten ist. Dadurch muss man keine Terminal oder Commands benutzten.
Natürlich müssen die geforderten Liabrys vorhanden sein.

## 👥 Die Autoren

Projektarbeit von:
* **Emre Kaplan**
* **Bünyamin Budak**
* **Maxim Speczyk**

*Hochschule Düsseldorf - Fachbereich Elektro- und Informationstechnik*
