import warnings

warnings.filterwarnings('ignore')  # Ignora todos os avisos

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.prever import classificar_imagem
from utils.config import local_input, local_output, local_output_errados
from utils.listar_imagens import listar_imagens
from utils.salvar_imagens import salvar_imagem


def main():
    # Caminho do diretório que contém as imagens

    # Lista todas as imagens no diretório
    imagens = listar_imagens(local_input)

    # Itera sobre cada imagem e classifica
    contagem = 0
    for imagem in imagens:
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
