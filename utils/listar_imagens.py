import os
from pathlib import Path

from utils.config import extensoes_imagem

def listar_imagens(caminho):
    """
    Lista todas as imagens encontradas no diretório especificado.
    
    Args:
        caminho (str): Caminho do diretório que contém as imagens
    
    Returns:
        list: Lista dos caminhos de arquivos de imagens
    """  
    
    # Cria um objeto Path a partir do caminho do diretório
    diretorio = Path(caminho)
    
    # Verifica se o diretório existe
    if not diretorio.exists() or not diretorio.is_dir():
        raise ValueError(f"O caminho especificado '{caminho}' não existe ou não é um diretório.")
    
    # Lista para armazenar caminhos de arquivos de imagem
    arquivos_imagem = []
    
    # Itera por todos os arquivos no diretório
    for arquivo in diretorio.iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes_imagem:
            arquivos_imagem.append(str(arquivo))
    
    return arquivos_imagem

if __name__ == "__main__":

    caminho = r"E:\Py_Projetos\FrameQC\tests\inputs"

    print(listar_imagens(caminho))