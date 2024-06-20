import tkinter as tk
from tkinter import ttk, scrolledtext
from ttkthemes import ThemedTk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2Tokenizer, GPT2LMHeadModel
import torch

class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sélecteur de Modèle")
        self.root.geometry("800x600")

        self.model_option = tk.StringVar()
        self.model_option.set("flan_t5_shrump")

        # Liste des modèles disponibles
        self.models = ["flan_t5_shrump", "gpt2_shrump"]

        self.create_widgets()
        self.tokenizer = None
        self.model = None
        self.is_custom_model = False

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')  # Thème pour une meilleure personnalisation

        # Configurer les styles pour les widgets
        style.configure('TLabel', font=('Helvetica', 14))
        style.configure('TButton', font=('Helvetica', 14), padding=6)
        style.map('TButton', background=[('active', '#FFA04D')])
        style.configure('TCombobox', font=('Helvetica', 14))

        main_frame = ttk.Frame(self.root, padding="20 20 20 20", style='TFrame')
        main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(2, weight=1)

        ttk.Label(main_frame, text="Sélectionnez un modèle :").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)

        self.model_combo = ttk.Combobox(main_frame, textvariable=self.model_option, values=self.models)
        self.model_combo.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        self.load_button = ttk.Button(main_frame, text="Charger le modèle", command=self.load_model, style='TButton')
        self.load_button.grid(column=2, row=0, padx=10, pady=10)

        self.prompt_label = ttk.Label(main_frame, text="Entrez votre prompt :")
        self.prompt_label.grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.prompt_entry = ttk.Entry(main_frame, width=50, font=('Helvetica', 14))
        self.prompt_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

        self.generate_button = ttk.Button(main_frame, text="Générer du texte", command=self.generate_text, style='TButton')
        self.generate_button.grid(column=2, row=1, padx=10, pady=10)

        self.output_text = scrolledtext.ScrolledText(main_frame, width=80, height=20, font=('Helvetica', 14), wrap=tk.WORD, borderwidth=2, relief="solid")
        self.output_text.grid(column=0, row=2, columnspan=3, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

    def load_model(self):
        model_name = self.model_option.get()
        if model_name == "gpt2_shrump":
            # Charger le modèle GPT2 personnalisé
            self.tokenizer = GPT2Tokenizer.from_pretrained(r'C:\Users\marco\OneDrive\Desktop\TXTgeneration\GPT2\gpt2_shrump')
            self.model = GPT2LMHeadModel.from_pretrained(r'C:\Users\marco\OneDrive\Desktop\TXTgeneration\GPT2\gpt2_shrump')
            self.is_custom_model = True
            self.prompt_label.grid_remove()
            self.prompt_entry.grid_remove()
        elif model_name == "flan_t5_shrump":
            # Charger le modèle Flan-T5 personnalisé
            self.tokenizer = AutoTokenizer.from_pretrained(r'C:\Users\marco\OneDrive\Desktop\TXTgeneration\flan_10percent')
            self.model = AutoModelForSeq2SeqLM.from_pretrained(r'C:\Users\marco\OneDrive\Desktop\TXTgeneration\flan_10percent')
            self.is_custom_model = False
            self.prompt_label.grid()
            self.prompt_entry.grid()
        else:
            # Charger un modèle par défaut
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_custom_model = False
            self.prompt_label.grid()
            self.prompt_entry.grid()
        self.output_text.insert(tk.END, f"Modèle {model_name} chargé avec succès !\n")

    def generate_text(self):
        if self.model and self.tokenizer:
            if self.is_custom_model:
                prompt = self.prompt_entry.get()
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
                outputs = self.model.generate(
                    input_ids,
                    max_length=100,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                self.output_text.insert(tk.END, "Shrump: ", "bold")
                self.output_text.insert(tk.END, f"{generated_text}\n")
            else:
                prompt = self.prompt_entry.get()
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                self.output_text.insert(tk.END, f"{prompt}\n")
                self.output_text.insert(tk.END, "Shrump: ", "bold")
                self.output_text.insert(tk.END, f"{generated_text}\n")
        else:
            self.output_text.insert(tk.END, "Veuillez d'abord charger un modèle !\n")

if __name__ == "__main__":
    root = ThemedTk(theme="radiance")  # Utiliser le thème 'radiance'

    # Configurer la balise pour le texte en gras
    app = ModelApp(root)
    app.output_text.tag_configure("bold", font=('Helvetica', 14, 'bold'))

    root.mainloop()

