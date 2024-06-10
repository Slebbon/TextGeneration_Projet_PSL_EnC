import tkinter as tk
from tkinter import ttk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

class TextGenerationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Générateur de Texte")
        
        # Carica il tokenizer GPT-3
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        except Exception as e:
            print("Errore durante il caricamento del tokenizer:", e)

        # Carica il modello GPT-3
        try:
            self.model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-125M")
        except Exception as e:
            print("Errore durante il caricamento del modello:", e)

        self.create_widgets()

    def create_widgets(self):
        self.prompt_label = ttk.Label(self.master, text="Saisie :")
        self.prompt_label.grid(row=0, column=0, padx=5, pady=5)

        self.prompt_entry = ttk.Entry(self.master, width=50)
        self.prompt_entry.grid(row=0, column=1, padx=5, pady=5)

        self.generate_button = ttk.Button(self.master, text="Générer du texte", command=self.generate_text)
        self.generate_button.grid(row=0, column=2, padx=5, pady=5)

        self.output_text = tk.Text(self.master, width=70, height=20)
        self.output_text.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

    def generate_text(self):
        prompt = self.prompt_entry.get()
        generated_text = self._generate_text(prompt)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, generated_text)

    def _generate_text(self, prompt):
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            return "Modello o tokenizer non caricati correttamente."
        
        try:
            # Utilizza il modello per generare il testo
            text_generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            generated_text = text_generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            return generated_text
        except Exception as e:
            return "Errore durante la generazione del testo: " + str(e)

def main():
    root = tk.Tk()
    app = TextGenerationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

