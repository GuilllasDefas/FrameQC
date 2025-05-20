import os
from PIL import Image

def salvar_imagem(caminho_imagem, porcentagem, local_destino):
    """
    Salva uma cópia de uma imagem com o nome modificado para incluir a porcentagem.
    
    Args:
        caminho_imagem (str): Caminho da imagem original
        porcentagem (float ou int): Porcentagem a ser incluída no nome do arquivo
        local_destino (str): Diretório onde a imagem será salva
    
    Returns:
        str: Caminho completo da imagem salva
    """
    try:

        if not os.path.exists(local_destino):
            os.makedirs(local_destino)
            
        # Carregar a imagem
        imagem = Image.open(caminho_imagem)
        
        # Extrair nome do arquivo e extensão
        nome_arquivo = os.path.basename(caminho_imagem)
        nome_sem_extensao, extensao = os.path.splitext(nome_arquivo)
        
        # Criar novo nome com a porcentagem
        novo_nome = f"{porcentagem*100:.0f}%_{nome_sem_extensao}{extensao}"
        
        # Caminho completo para salvar
        caminho_destino = os.path.join(local_destino, novo_nome)
        
        # Salvar a imagem
        imagem.save(caminho_destino)
        
        return
    
    except FileNotFoundError as e:
        mensagem = f"Erro: Arquivo de imagem não encontrado: {e}"
        print(mensagem)
        return
    except PermissionError as e:
        mensagem = f"Erro: Sem permissão para acessar ou salvar a imagem: {e}"
        print(mensagem)
        return
    except OSError as e:
        mensagem = f"Erro do sistema operacional ao processar a imagem: {e}"
        print(mensagem)
        return
    except IOError as e:
        mensagem = f"Erro de entrada/saída ao processar a imagem: {e}"
        print(mensagem)
        return
    except Exception as e:
        mensagem = f"Erro inesperado ao processar a imagem: {e}"
        print(mensagem)
        return
    
if __name__ == "__main__":
    # TESTE
    caminho_imagem = r"E:\Py_Projetos\FrameQC\tests\inputs\frame_000120-crop.png"
    porcentagem = 75
    local_destino = r"E:\Py_Projetos\FrameQC\tests\outputs"

    caminho_salvo, mensagem = salvar_imagem(caminho_imagem, porcentagem, local_destino)

    if caminho_salvo:
        print(f"Imagem salva em: {caminho_salvo}")
    else:
        print(mensagem)