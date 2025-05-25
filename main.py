import warnings

warnings.filterwarnings('ignore')  # Ignora todos os avisos

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '0'  # Mudei de '1' para '0' para melhor performance GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from src.prever import classificar_imagem
from src.prever_CNN import predizer_imagem
from utils.config import local_input, local_output, local_output_errados
from utils.listar_imagens import listar_imagens
from utils.salvar_imagens import salvar_imagem



def main():
    # Caminho do diretório que contém as imagens

    # Lista todas as imagens no diretório
    imagens = listar_imagens(local_input)

    # Itera sobre cada imagem e classifica
    contagem = 0
    
    pergunta = input('Qual modelo deseja usar? CNN ("C") ou MobileNetV2 ("M")?: ')
    
    for imagem in imagens:
        if pergunta.lower() == 'c':
            prob = predizer_imagem(imagem)
        elif pergunta.lower() == 'm':
            prob = classificar_imagem(imagem)
        else:
            print("Opção inválida. Usando CNN por padrão.")
            prob = classificar_imagem(imagem)

    
        if prob >= 0.5:
            contagem += 1
            print(f'Imagems classificadas {contagem} de {len(imagens)}')
            salvar_imagem(imagem, prob, local_output)
        else:
            contagem += 1
            print(f'Imagems classificadas {contagem} de {len(imagens)}')
            salvar_imagem(imagem, prob, local_output_errados)


if __name__ == '__main__':
    main()
