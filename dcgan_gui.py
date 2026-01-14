import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import sys
import os
import glob
import re
import time
import shutil

# Laden der Backend-Logik für DCGAN
try:
    import dcgan_train_and_generate as logic
except ImportError:
    print("Fehler: 'dcgan_train_and_generate.py' wurde im Verzeichnis nicht gefunden.")
    sys.exit(1)

class DCGANApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DCGAN Studio v4")
        self.root.geometry("800x820")

        # UI Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
        self.style.configure("blue.Horizontal.TProgressbar", foreground='blue', background='#3498db')
        self.style.configure("Red.TButton", foreground="red")

        # Konfigurations-Variablen
        self.dataset_var = tk.StringVar(value="cifar10")
        self.epochs_var = tk.IntVar(value=20)
        self.batch_size_var = tk.IntVar(value=128)
        self.gen_count_var = tk.IntVar(value=1000)
        self.device_var = tk.StringVar(value="cpu")
        self.status_var = tk.StringVar(value="Bereit")

        # Zeit- und Fortschritts-Tracking
        self.time_info_var = tk.StringVar(value="Dauer letzte Epoche: --:--")
        self.last_epoch_time = 0
        self.current_epoch_idx = 0

        self.create_widgets()

        # Umleitung von stdout/stderr in das Log-Fenster
        self.redirector = TextRedirector(self)
        sys.stdout = self.redirector
        sys.stderr = self.redirector

    def create_widgets(self):
        # Steuerungspanel (links)
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side="left", fill="y")

        ttk.Label(control_frame, text="Einstellungen", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        # Datensatz-Auswahl
        group_ds = ttk.LabelFrame(control_frame, text="Datensatz", padding=5)
        group_ds.pack(fill="x", pady=5)
        ttk.Radiobutton(group_ds, text="CIFAR-10 (RGB)", variable=self.dataset_var, value="cifar10").pack(anchor="w")
        ttk.Radiobutton(group_ds, text="MNIST (Graustufen)", variable=self.dataset_var, value="mnist").pack(anchor="w")

        # Training-Parameter
        group_param = ttk.LabelFrame(control_frame, text="Hyperparameter", padding=5)
        group_param.pack(fill="x", pady=5)

        ttk.Label(group_param, text="Epochen:").pack(anchor="w")
        ttk.Entry(group_param, textvariable=self.epochs_var).pack(fill="x", pady=(0, 5))

        ttk.Label(group_param, text="Batch Size:").pack(anchor="w")
        ttk.Entry(group_param, textvariable=self.batch_size_var).pack(fill="x", pady=(0, 5))

        ttk.Label(group_param, text="Hardware:").pack(anchor="w")
        ttk.Combobox(group_param, textvariable=self.device_var, values=["cpu", "cuda"], state="readonly").pack(fill="x")

        self.btn_train = ttk.Button(control_frame, text="Training starten", command=self.start_training_thread)
        self.btn_train.pack(fill="x", pady=10)

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        # Bild-Generierung
        group_gen = ttk.LabelFrame(control_frame, text="Inferenz", padding=5)
        group_gen.pack(fill="x", pady=5)
        ttk.Label(group_gen, text="Anzahl Bilder:").pack(anchor="w")
        ttk.Entry(group_gen, textvariable=self.gen_count_var).pack(fill="x")

        self.btn_gen = ttk.Button(group_gen, text="Bilder generieren", command=self.start_gen_thread)
        self.btn_gen.pack(fill="x", pady=10)

        self.btn_reset = ttk.Button(control_frame, text="Verzeichnisse leeren", style="Red.TButton",
                                    command=self.reset_project)
        self.btn_reset.pack(fill="x", side="bottom", pady=10)

        # Anzeige-Bereich (rechts)
        display_frame = ttk.Frame(self.root, padding=10)
        display_frame.pack(side="right", fill="both", expand=True)

        top_info_frame = ttk.Frame(display_frame)
        top_info_frame.pack(fill="x")
        ttk.Label(top_info_frame, text="Status:", font=("Arial", 10, "bold")).pack(side="left")
        ttk.Label(top_info_frame, textvariable=self.status_var, foreground="blue").pack(side="left", padx=5)
        ttk.Label(top_info_frame, textvariable=self.time_info_var, foreground="red", font=("Arial", 9, "bold")).pack(
            side="right")

        # Fortschrittsbalken für Epochen und Batches
        ttk.Label(display_frame, text="Gesamtfortschritt (Epochen):", font=("Arial", 8)).pack(anchor="w", pady=(10, 0))
        self.progress_total = ttk.Progressbar(display_frame, orient="horizontal", mode="determinate",
                                              style="green.Horizontal.TProgressbar")
        self.progress_total.pack(fill="x", pady=2)

        ttk.Label(display_frame, text="Aktueller Batch:", font=("Arial", 8)).pack(anchor="w", pady=(5, 0))
        self.progress_epoch = ttk.Progressbar(display_frame, orient="horizontal", mode="determinate",
                                              style="blue.Horizontal.TProgressbar")
        self.progress_epoch.pack(fill="x", pady=2)

        # Bild-Vorschau
        self.img_label = ttk.Label(display_frame, text="Warten auf Trainingsstart...", relief="sunken", anchor="center")
        self.img_label.pack(fill="both", expand=True, pady=10)

        # Konsolen-Ausgabe
        ttk.Label(display_frame, text="Protokoll:").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(display_frame, state='disabled', height=8, font=("Consolas", 8))
        self.log_area.pack(fill="x")

    def toggle_buttons(self, state):
        """Aktiviert oder deaktiviert Buttons während Prozessen."""
        s = "normal" if state else "disabled"
        self.btn_train.config(state=s)
        self.btn_gen.config(state=s)
        self.btn_reset.config(state=s)

    def reset_project(self):
        """Löscht Checkpoints und generierte Bilder nach Bestätigung."""
        if messagebox.askyesno("Sicherheitsabfrage", "Sollen alle Modelle und Bilder gelöscht werden?"):
            try:
                for path in [logic.CHECKPOINT_DIR, logic.SAMPLES_DIR]:
                    if os.path.exists(path): shutil.rmtree(path)
                    os.makedirs(path, exist_ok=True)
                self.img_label.config(image='', text="Daten bereinigt")
                self.status_var.set("Reset durchgeführt.")
                self.progress_total['value'] = 0
                self.progress_epoch['value'] = 0
            except Exception as e:
                messagebox.showerror("Fehler", f"Löschvorgang fehlgeschlagen: {e}")

    def update_image_preview(self, filepath):
        """Skaliert und zeigt das aktuellste Vorschaubild an."""
        try:
            load = Image.open(filepath)
            load.thumbnail((500, 500))
            render = ImageTk.PhotoImage(load)
            self.img_label.config(image=render, text="")
            self.img_label.image = render
        except Exception as e:
            print(f"Fehler beim Laden der Vorschau: {e}")

    def find_latest_image(self, dataset):
        """Sucht die zuletzt gespeicherte Bilddatei des Datensatzes."""
        search_pattern = os.path.join(logic.SAMPLES_DIR, f"{dataset}_*.png")
        list_of_files = glob.glob(search_pattern)
        if list_of_files:
            return max(list_of_files, key=os.path.getctime)
        return None

    def start_training_thread(self):
        """Initialisiert das Training in einem separaten Thread."""
        self.progress_total['value'] = 0
        self.progress_epoch['value'] = 0
        self.status_var.set("Initialisierung...")
        self.last_epoch_time = time.time()
        self.current_epoch_idx = 0
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        """Führt die Trainingsschleife der Logik aus."""
        self.toggle_buttons(False)
        try:
            ds = self.dataset_var.get()
            ep = self.epochs_var.get()
            self.progress_total['maximum'] = ep

            logic.train_dcgan(dataset=ds, epochs=ep, batch_size=self.batch_size_var.get(), device=self.device_var.get())

            self.status_var.set("Training beendet")
            latest = self.find_latest_image(ds)
            if latest: self.root.after(0, self.update_image_preview, latest)
            messagebox.showinfo("Erfolg", "Das Training wurde erfolgreich abgeschlossen.")
        except Exception as e:
            messagebox.showerror("Trainingsfehler", str(e))
        finally:
            self.toggle_buttons(True)

    def start_gen_thread(self):
        """Startet die Bildgenerierung in einem Hintergrund-Thread."""
        self.status_var.set("Generierung läuft...")
        threading.Thread(target=self.run_gen, daemon=True).start()

    def run_gen(self):
        """Lädt Checkpoint und generiert neue Samples."""
        self.toggle_buttons(False)
        try:
            ds = self.dataset_var.get()
            ckpt = os.path.join(logic.CHECKPOINT_DIR, f"generator_{ds}.pth")
            if not os.path.exists(ckpt):
                messagebox.showwarning("Hinweis", "Kein trainiertes Modell gefunden.")
                return

            logic.generate_images_from_checkpoint(ckpt, dataset=ds, n_samples=self.gen_count_var.get(),
                                                  device=self.device_var.get())
            self.status_var.set("Generierung abgeschlossen")

            preview_path = os.path.join(logic.SAMPLES_DIR, f"{ds}_gan_preview_{self.gen_count_var.get()}.png")
            if os.path.exists(preview_path):
                self.root.after(0, self.update_image_preview, preview_path)
        except Exception as e:
            print(f"Fehler bei Generierung: {e}")
        finally:
            self.toggle_buttons(True)

    def update_epoch_progress(self, current, total):
        """Aktualisiert den unteren Fortschrittsbalken (Batches)."""
        self.root.after(0, lambda: self.progress_epoch.configure(maximum=total, value=current))

    def update_total_progress(self, current_ep, total_ep):
        """Berechnet Epochendauer und aktualisiert den oberen Balken."""
        if current_ep > self.current_epoch_idx:
            now = time.time()
            if self.current_epoch_idx > 0:
                duration = now - self.last_epoch_time
                mins, secs = divmod(duration, 60)
                self.root.after(0, lambda: self.time_info_var.set(f"Letzte Epoche: {int(mins)}m {int(secs)}s"))
            self.last_epoch_time = now
            self.current_epoch_idx = current_ep

        self.root.after(0, lambda: self.progress_total.configure(maximum=total_ep, value=current_ep))

        # Automatisches Update der Vorschau bei jeder neuen Epoche
        ds = self.dataset_var.get()
        img_path = os.path.join(logic.SAMPLES_DIR, f"{ds}_epoch{current_ep:03d}.png")
        if os.path.exists(img_path):
            self.root.after(0, self.update_image_preview, img_path)


class TextRedirector(object):
    """Leitet Konsolenausgaben an das Tkinter ScrolledText Widget weiter."""
    def __init__(self, app):
        self.app = app

    def write(self, string):
        # Filtert Epochen-Informationen aus dem Stream (Regex-Abgleich)
        match_epoch = re.search(r"Epoch (\d+)/(\d+)", string)
        if match_epoch:
            self.app.update_total_progress(int(match_epoch.group(1)), int(match_epoch.group(2)))

        # Extrahiert Batch-Fortschritt (z.B. "10/100")
        if "Epoch" not in string:
            match_batch = re.search(r"(\d+)/(\d+)", string)
            if match_batch:
                try:
                    cur, tot = int(match_batch.group(1)), int(match_batch.group(2))
                    if cur <= tot:
                        self.app.update_epoch_progress(cur, tot)
                        self.app.status_var.set(f"Batch {cur}/{tot}")
                except: pass

        # Entfernt Escape-Sequenzen und tqmd-Reste aus dem Log-Fenster
        if "\r" in string or "%" in string:
            return

        clean = string.strip()
        if clean:
            self.app.log_area.after(0, self._append, string)

    def _append(self, string):
        self.app.log_area.configure(state='normal')
        self.app.log_area.insert("end", string)
        self.app.log_area.see("end")
        self.app.log_area.configure(state='disabled')

    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = DCGANApp(root)
    root.mainloop()