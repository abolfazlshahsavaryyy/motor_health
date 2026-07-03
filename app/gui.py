import tkinter as tk
from tkinter import filedialog, messagebox
from service.model.load_models import load_models
from service.audio.read_audio import load_single_audio
from service.processing.audio_processing import extract_mfcc

class MotorHealthGUI:

    def __init__(self):

        self.window = tk.Tk()

        self.window.title("Motor Health Detection")

        self.window.geometry("600x350")

        self.window.resizable(False, False)

        self.selected_file = None

        self.models = load_models()

        self.create_widgets()

    def create_widgets(self):

        title = tk.Label(
            self.window,
            text="Motor Health Detection",
            font=("Arial", 18, "bold")
        )

        title.pack(pady=20)

        self.file_label = tk.Label(
            self.window,
            text="No audio selected",
            wraplength=500
        )

        self.file_label.pack(pady=10)

        browse_button = tk.Button(
            self.window,
            text="Choose Audio",
            width=20,
            command=self.choose_audio
        )

        browse_button.pack(pady=10)

        process_button = tk.Button(
            self.window,
            text="Process Audio",
            width=20,
            command=self.process_audio
        )

        process_button.pack(pady=10)

    def choose_audio(self):

        file_path = filedialog.askopenfilename(

            title="Select Audio File",

            filetypes=[
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )

        if file_path:

            self.selected_file = file_path

            self.file_label.config(text=file_path)

    def process_audio(self):

        if self.selected_file is None:

            messagebox.showwarning(
                "Warning",
                "Please select an audio file first."
            )
            return

        audio, sr = load_single_audio(self.selected_file)

        mfcc = extract_mfcc(
            audio,
            sr,
            n_mfcc=13
        )

        features = mfcc.flatten()

        features = features.reshape(1, -1)

        label_names = {
            1: "Healthy",
            2: "Bearing",
            3: "Propeller"
        }

        results = {}

        for model_name, model in self.models.items():
        
            prediction = model.predict(features)[0]

            results[model_name] = label_names[prediction]

        print(results)

        message = "Prediction Results\n\n"

        for model_name, prediction in results.items():
            message += f"{model_name:22}: {prediction}\n"

        messagebox.showinfo(
            "Prediction Results",
            message
        )
    def run(self):

        self.window.mainloop()


def run_app():

    app = MotorHealthGUI()

    app.run()