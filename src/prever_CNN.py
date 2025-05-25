import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constantes
TARGET_SIZE = (224, 224)  # Mesmo tamanho usado no treinamento
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
    img_tensor = tf.expand_dims(img_tensor, axis=-1)  # Adiciona canal
    img_tensor = tf.expand_dims(img_tensor, axis=0)   # Adiciona batch
    
    # Aplica Sobel filter
    sobel = tf.image.sobel_edges(img_tensor)
    
    # Reformata para ter 2 canais (gx, gy) - formato consistente com o treinamento
    sobel = tf.reshape(sobel, [-1, TARGET_SIZE[0], TARGET_SIZE[1], 2])
    
    # Normaliza os valores
    sobel = sobel / (tf.reduce_max(tf.abs(sobel)) + 1e-9) * 0.5 + 0.5
    
    return sobel

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
