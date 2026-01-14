# 🕵️‍♂️ Clustering: MNIST vs. CIFAR-10

Willkommen in unserem Repo! Dieses Projekt entstand im Rahmen unseres Mastermoduls "Künstliche Intelligenz".

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

## Selbst ausprobieren
Wir haben ein kleines GUI programmiert, so dass der Code einfach zu benutzten ist. Dadurch muss man keine Terminal oder Commands benutzten.
[Hier geht es zur DCGAN GUI](https://github.com/MaximSpeczyk/KI-Projekt-Hochschule-Duesseldorf-WS25-26/blob/main/Train_Generate/dcgan_gui.py)
Natürlich müssen die geforderten Liabrys vorhanden sein.

## 👥 Die Autoren

Projektarbeit von:
* **Emre Kaplan**
* **Bünyamin Budak**
* **Maxim Speczyk**

*Hochschule Düsseldorf - Fachbereich Elektro- und Informationstechnik*
