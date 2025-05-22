#Servirá apenas para armazenar as variáveis de configuração do projeto
# e facilitar a manutenção do código.
import os

extensoes_imagem = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

local_input = r"tests\inputs"

local_output = os.path.join(local_input, "Certos")
local_output_errados = os.path.join(local_input, "Errados")

#local_input = r"tests\inputs"
#local_output = r"tests\outputs\certos"
#local_output_errados = r"tests\outputs\errados"
