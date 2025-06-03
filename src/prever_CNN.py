import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constantes
TARGET_SIZE = (320, 320)  # Mesmo tamanho usado no treinamento
MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'cnn', 'best_cnn_adapt.keras')

# Verifica disponibilidade de GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) detectada(s): {len(gpus)}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")
else:
    print("Nenhuma GPU detectada. Usando CPU.")

# Carrega o modelo
modelo = None

def preprocessa_imagem(caminho_imagem):
    """Prepara a imagem para predição usando o mesmo processamento do treinamento"""
    # Carrega a imagem em escala de cinza
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")
    
    # Redimensiona para o tamanho alvo
    img_redimensionada = cv2.resize(img, TARGET_SIZE)
    
    # Converte para tensor e ajusta dimensões para sobel
    img_tensor = tf.convert_to_tensor(img_redimensionada, dtype=tf.float32)
    img_tensor = img_tensor / 255.0  # Normalização melhorada
    img_tensor = tf.expand_dims(img_tensor, axis=-1)  # Adiciona canal
    img_tensor = tf.expand_dims(img_tensor, axis=0)   # Adiciona batch
    
    # Aplica Sobel filter
    sobel = tf.image.sobel_edges(img_tensor)
    
    # Extrai gx e gy
    gx = sobel[0, :, :, :, 0]
    gy = sobel[0, :, :, :, 1]
    
    # Calcula a magnitude
    magnitude = tf.sqrt(tf.square(gx) + tf.square(gy))
    
    # Adicionar filtro Laplaciano para detectar bordas de segunda ordem (exatamente como no treino)
    kernel_laplaciano = tf.constant([[[[-1]], [[-1]], [[-1]]],
                                  [[[-1]], [[8]], [[-1]]],
                                  [[[-1]], [[-1]], [[-1]]]], dtype=tf.float32)
    laplacian = tf.nn.conv2d(img_tensor, kernel_laplaciano, strides=[1,1,1,1], padding='SAME')
    laplacian = tf.clip_by_value(laplacian, -1, 1) * 0.5 + 0.5
    laplacian = laplacian[0]  # Remove a dimensão do batch
    
    # Combina os 4 canais: Sobel_x, Sobel_y, Magnitude, Laplacian - mesma ordem do treino
    combinado = tf.concat([gx, gy, magnitude, laplacian], axis=-1)
    
    # Adiciona a dimensão do batch
    combinado = tf.expand_dims(combinado, axis=0)
    
    return combinado

def predizer_imagem(caminho_imagem):
    """Prediz se a imagem é válida ou não usando o modelo CNN"""
    global modelo
    
    # Carrega o modelo se ainda não foi carregado
    if modelo is None:
        modelo = load_model(MODELO_PATH)
        print(f"Modelo carregado de: {MODELO_PATH}")
    
    # Pré-processa a imagem
    tensor = preprocessa_imagem(caminho_imagem)
    
    # Realiza a predição
    predicao = modelo.predict(tensor, verbose=0)[0][0]
    
    return float(predicao)
