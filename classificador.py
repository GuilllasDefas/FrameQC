import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ClassificadorImagens:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Imagens")
        self.root.geometry("1200x900")
        # Definir cor de fundo escura
        self.bg_color = "#23272e"
        self.fg_color = "#f0f0f0"
        self.button_certo_color = "#2ecc40"
        self.button_errado_color = "#e74c3c"
        self.button_hover = "#555"
        self.status_bg = "#181a1b"
        self.label_img_bg = "#181a1b"
        self.label_img_fg = "#ffd700"
        # Variáveis
        self.pasta_origem = ""
        self.imagem_atual_caminho = None
        self.arquivos_imagem = []
        self.indice_atual = 0

        self.configurar_interface()

        # Atalhos de teclado
        self.root.bind('<Right>', lambda event: self.mover_imagem("certo") if self.botao_certo['state'] == tk.NORMAL else None)
        self.root.bind('<Return>', lambda event: self.mover_imagem("certo") if self.botao_certo['state'] == tk.NORMAL else None)
        self.root.bind('<Left>', lambda event: self.mover_imagem("errado") if self.botao_errado['state'] == tk.NORMAL else None)
        self.root.bind('<Escape>', lambda event: self.mover_imagem("errado") if self.botao_errado['state'] == tk.NORMAL else None)
        self.root.bind('<space>', lambda event: self.pular_imagem() if self.botao_pular['state'] == tk.NORMAL else None)
        self.root.bind('<Shift-space>', lambda event: self.pular_10_imagens() if self.botao_pular10['state'] == tk.NORMAL else None)
        self.root.bind('<BackSpace>', lambda event: self.voltar_imagem() if self.botao_voltar['state'] == tk.NORMAL else None)
        self.frame_imagem.bind('<Configure>', lambda event: self.exibir_imagem_atual(redimensionar=True))

    def configurar_interface(self):
        # Label do nome da imagem acima da imagem
        self.label_nome_imagem = tk.Label(
            self.root, text="", font=("Arial", 16, "bold"),
            bg=self.label_img_bg, fg=self.label_img_fg, pady=10
        )
        self.label_nome_imagem.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(10, 0))

        # Frame para exibição da imagem
        self.frame_imagem = tk.Frame(self.root, bg=self.bg_color)
        self.frame_imagem.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 10))

        # Label para exibir a imagem
        self.label_imagem = tk.Label(self.frame_imagem, bg=self.bg_color)
        self.label_imagem.pack(fill=tk.BOTH, expand=True)

        # Frame para botões centralizados abaixo da imagem
        self.frame_botoes = tk.Frame(self.root, bg=self.bg_color)
        self.frame_botoes.pack(fill=tk.X, padx=30, pady=(0, 25))

        # Botão Voltar
        self.botao_voltar = tk.Button(
            self.frame_botoes, text="⟵ Voltar\n[Backspace]", bg="#444", fg=self.fg_color,
            command=self.voltar_imagem, state=tk.DISABLED,
            activebackground=self.button_hover, activeforeground=self.fg_color,
            font=("Arial", 14, "bold"), height=2, width=12, relief=tk.RAISED, bd=2
        )
        self.botao_voltar.pack(side=tk.LEFT, padx=12, pady=0, expand=True)

        # Botão de selecionar pasta
        self.botao_selecionar = tk.Button(
            self.frame_botoes, text="Selecionar Pasta", command=self.selecionar_pasta,
            bg=self.bg_color, fg=self.fg_color, activebackground=self.button_hover, activeforeground=self.fg_color,
            font=("Arial", 14, "bold"), height=2, width=16, relief=tk.RAISED, bd=2
        )
        self.botao_selecionar.pack(side=tk.LEFT, padx=12, pady=0, expand=True)

        # Botão Certo
        self.botao_certo = tk.Button(
            self.frame_botoes, text="✔ Certo\n[Enter/→]", bg=self.button_certo_color, fg="white",
            command=lambda: self.mover_imagem("certo"), state=tk.DISABLED,
            activebackground="#27ae60", activeforeground="white",
            font=("Arial", 14, "bold"), height=2, width=12, relief=tk.RAISED, bd=2
        )
        self.botao_certo.pack(side=tk.LEFT, padx=12, pady=0, expand=True)

        # Botão Errado
        self.botao_errado = tk.Button(
            self.frame_botoes, text="✖ Errado\n[Esc/←]", bg=self.button_errado_color, fg="white",
            command=lambda: self.mover_imagem("errado"), state=tk.DISABLED,
            activebackground="#c0392b", activeforeground="white",
            font=("Arial", 14, "bold"), height=2, width=12, relief=tk.RAISED, bd=2
        )
        self.botao_errado.pack(side=tk.LEFT, padx=12, pady=0, expand=True)

        # Botão Pular
        self.botao_pular = tk.Button(
            self.frame_botoes, text="Pular\n[Espaço]", bg="#444", fg=self.fg_color,
            command=self.pular_imagem, state=tk.DISABLED,
            activebackground=self.button_hover, activeforeground=self.fg_color,
            font=("Arial", 14, "bold"), height=2, width=12, relief=tk.RAISED, bd=2
        )
        self.botao_pular.pack(side=tk.LEFT, padx=12, pady=0, expand=True)

        # Botão Pular 10
        self.botao_pular10 = tk.Button(
            self.frame_botoes, text="Pular 10\n[Shift+Espaço]", bg="#444", fg=self.fg_color,
            command=self.pular_10_imagens, state=tk.DISABLED,
            activebackground=self.button_hover, activeforeground=self.fg_color,
            font=("Arial", 14, "bold"), height=2, width=12, relief=tk.RAISED, bd=2
        )
        self.botao_pular10.pack(side=tk.LEFT, padx=12, pady=0, expand=True)

        # Label de status
        self.label_status = tk.Label(
            self.root, text="Selecione uma pasta para começar a classificar imagens",
            bd=1, relief=tk.SUNKEN, anchor=tk.W,
            bg=self.status_bg, fg=self.fg_color, font=("Arial", 12)
        )
        self.label_status.pack(side=tk.BOTTOM, fill=tk.X)

    def selecionar_pasta(self):
        self.pasta_origem = filedialog.askdirectory(title="Selecionar Pasta com Imagens")
        if not self.pasta_origem:
            return
        os.makedirs(os.path.join(self.pasta_origem, "certo"), exist_ok=True)
        os.makedirs(os.path.join(self.pasta_origem, "errado"), exist_ok=True)
        self.arquivos_imagem = [f for f in os.listdir(self.pasta_origem)
                             if os.path.isfile(os.path.join(self.pasta_origem, f)) and
                             f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not self.arquivos_imagem:
            messagebox.showinfo("Sem Imagens", "Nenhuma imagem encontrada na pasta selecionada.")
            return
        self.botao_certo.config(state=tk.NORMAL)
        self.botao_errado.config(state=tk.NORMAL)
        self.botao_pular.config(state=tk.NORMAL)
        self.botao_pular10.config(state=tk.NORMAL)
        self.botao_voltar.config(state=tk.NORMAL)
        self.indice_atual = 0
        self.exibir_imagem_atual()

    def exibir_imagem_atual(self, redimensionar=False):
        if not self.arquivos_imagem:
            self.label_imagem.config(image=None)
            self.label_nome_imagem.config(text="")
            self.atualizar_status_final()
            self.botao_certo.config(state=tk.DISABLED)
            self.botao_errado.config(state=tk.DISABLED)
            self.botao_pular.config(state=tk.DISABLED)
            self.botao_pular10.config(state=tk.DISABLED)
            self.botao_voltar.config(state=tk.DISABLED)
            return
        # Exibir nome da imagem acima da imagem
        nome_img = self.arquivos_imagem[self.indice_atual]
        self.label_nome_imagem.config(text=nome_img)
        # Exibir imagem atual
        self.imagem_atual_caminho = os.path.join(self.pasta_origem, nome_img)
        try:
            img = Image.open(self.imagem_atual_caminho)
            img = self.redimensionar_imagem(img)
            self.photo = ImageTk.PhotoImage(img)
            self.label_imagem.config(image=self.photo, bg=self.bg_color)
            self.atualizar_status()
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir a imagem: {str(e)}")
            self.arquivos_imagem.pop(self.indice_atual)
            if self.arquivos_imagem:
                if self.indice_atual >= len(self.arquivos_imagem):
                    self.indice_atual = 0
                self.exibir_imagem_atual()
            else:
                self.label_imagem.config(image=None)
                self.label_nome_imagem.config(text="")
                self.atualizar_status_final()
                self.botao_certo.config(state=tk.DISABLED)
                self.botao_errado.config(state=tk.DISABLED)
                self.botao_pular.config(state=tk.DISABLED)
                self.botao_pular10.config(state=tk.DISABLED)
                self.botao_voltar.config(state=tk.DISABLED)

    def redimensionar_imagem(self, img):
        width, height = img.size
        max_width = self.frame_imagem.winfo_width() - 20
        max_height = self.frame_imagem.winfo_height() - 20
        if max_width <= 0:
            max_width = 700
        if max_height <= 0:
            max_height = 500
        aspect_ratio = width / height
        if width > max_width:
            width = max_width
            height = int(width / aspect_ratio)
        if height > max_height:
            height = max_height
            width = int(height * aspect_ratio)
        try:
            return img.resize((width, height), Image.LANCZOS)
        except AttributeError:
            try:
                return img.resize((width, height), Image.ANTIALIAS)
            except AttributeError:
                return img.resize((width, height))

    def atualizar_status(self):
        n_certo = len([f for f in os.listdir(os.path.join(self.pasta_origem, "certo")) if os.path.isfile(os.path.join(self.pasta_origem, "certo", f))])
        n_errado = len([f for f in os.listdir(os.path.join(self.pasta_origem, "errado")) if os.path.isfile(os.path.join(self.pasta_origem, "errado", f))])
        self.label_status.config(
            text=f"Imagem {self.indice_atual + 1} de {len(self.arquivos_imagem)}: {self.arquivos_imagem[self.indice_atual]}   |   Certo: {n_certo}   Errado: {n_errado}"
        )

    def atualizar_status_final(self):
        n_certo = len([f for f in os.listdir(os.path.join(self.pasta_origem, "certo")) if os.path.isfile(os.path.join(self.pasta_origem, "certo", f))])
        n_errado = len([f for f in os.listdir(os.path.join(self.pasta_origem, "errado")) if os.path.isfile(os.path.join(self.pasta_origem, "errado", f))])
        self.label_status.config(
            text=f"Todas as imagens foram classificadas!   |   Certo: {n_certo}   Errado: {n_errado}"
        )

    def mover_imagem(self, destino):
        if not self.imagem_atual_caminho:
            return
        pasta_destino = os.path.join(self.pasta_origem, destino)
        nome_arquivo = os.path.basename(self.imagem_atual_caminho)
        caminho_destino = os.path.join(pasta_destino, nome_arquivo)
        try:
            shutil.move(self.imagem_atual_caminho, caminho_destino)
            self.arquivos_imagem.pop(self.indice_atual)
            if self.arquivos_imagem:
                if self.indice_atual >= len(self.arquivos_imagem):
                    self.indice_atual = 0
                self.exibir_imagem_atual()
            else:
                self.label_imagem.config(image=None)
                self.label_nome_imagem.config(text="")
                self.atualizar_status_final()
                self.botao_certo.config(state=tk.DISABLED)
                self.botao_errado.config(state=tk.DISABLED)
                self.botao_pular.config(state=tk.DISABLED)
                self.botao_pular10.config(state=tk.DISABLED)
                self.botao_voltar.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao mover o arquivo: {str(e)}")

    def pular_imagem(self):
        if not self.arquivos_imagem:
            return
        self.indice_atual += 1
        if self.indice_atual >= len(self.arquivos_imagem):
            self.indice_atual = 0
        self.exibir_imagem_atual()

    def pular_10_imagens(self):
        if not self.arquivos_imagem:
            return
        self.indice_atual += 10
        if self.indice_atual >= len(self.arquivos_imagem):
            self.indice_atual = self.indice_atual % len(self.arquivos_imagem)
        self.exibir_imagem_atual()

    def voltar_imagem(self):
        if not self.arquivos_imagem:
            return
        self.indice_atual -= 1
        if self.indice_atual < 0:
            self.indice_atual = len(self.arquivos_imagem) - 1
        self.exibir_imagem_atual()

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#23272e")
    app = ClassificadorImagens(root)
    root.mainloop()
