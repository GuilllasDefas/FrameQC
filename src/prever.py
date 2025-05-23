from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Carrega o modelo salvo
model = load_model('./src/models/FrameQC_model.keras')

def classificar_imagem(caminho_img):
    img = image.load_img(caminho_img, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)    # cria batch de tamanho 1
    prob = model.predict(arr)[0][0]      # probabilidade de "correta"
    return prob