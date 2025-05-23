#Servirá apenas para armazenar as variáveis de configuração do projeto
# e facilitar a manutenção do código.
import os
import sys
from tkinter import filedialog


extensoes_imagem = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

local_input = filedialog.askdirectory(title="Selecione o diretório de entrada")
if not local_input:
    sys.exit(0)  # Encerra o programa se nenhum diretório for selecionado
    raise ValueError("Nenhum diretório selecionado. O programa será encerrado.")





local_output = os.path.join(local_input, "Certos")
local_output_errados = os.path.join(local_input, "Errados")

#local_input = r"tests\inputs"
#local_output = r"tests\outputs\certos"
#local_output_errados = r"tests\outputs\errados"
