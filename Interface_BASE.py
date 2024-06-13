import tkinter as tk
from tkinter import ttk, scrolledtext
from ttkthemes import ThemedTk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Selector")
        self.root.geometry("800x600")

        self.model_option = tk.StringVar()
        self.model_option.set("google/flan-t5-base")

        self.models = ["google/flan-t5-base", "another-model-name", "yet-another-model-name"]

        self.create_widgets()
        self.tokenizer = None
        self.model = None

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')  # Thème qui permet une personnalisation accrue

        # Configuration des styles pour les différents widgets
        style.configure('TLabel', font=('Helvetica', 14))
        style.configure('TButton', font=('Helvetica', 14), padding=6)
        style.map('TButton', background=[('active', '#FFA04D')])
        style.configure('TCombobox', font=('Helvetica', 14))

        main_frame = ttk.Frame(self.root, padding="20 20 20 20", style='TFrame')
        main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(2, weight=1)

        ttk.Label(main_frame, text="Select a model:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)

        self.model_combo = ttk.Combobox(main_frame, textvariable=self.model_option, values=self.models)
        self.model_combo.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        self.load_button = ttk.Button(main_frame, text="Load Model", command=self.load_model, style='TButton')
        self.load_button.grid(column=2, row=0, padx=10, pady=10)

        ttk.Label(main_frame, text="Enter your prompt:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.prompt_entry = ttk.Entry(main_frame, width=50, font=('Helvetica', 14))
        self.prompt_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

        self.generate_button = ttk.Button(main_frame, text="Generate Text", command=self.generate_text, style='TButton')
        self.generate_button.grid(column=2, row=1, padx=10, pady=10)

        self.output_text = scrolledtext.ScrolledText(main_frame, width=80, height=20, font=('Helvetica', 14), wrap=tk.WORD, borderwidth=2, relief="solid")
        self.output_text.grid(column=0, row=2, columnspan=3, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

    def load_model(self):
        model_name = self.model_option.get()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.output_text.insert(tk.END, f"Model {model_name} loaded successfully!\n")

    def generate_text(self):
        if self.model and self.tokenizer:
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
            self.output_text.insert(tk.END, "Please load a model first!\n")

if __name__ == "__main__":
    root = ThemedTk(theme="radiance")  # Utiliser le thème moderne 'radiance'

    # Configuration du tag pour le texte en gras
    app = ModelApp(root)
    app.output_text.tag_configure("bold", font=('Helvetica', 14, 'bold'))

    root.mainloop()


