import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Carrega o modelo salvo
model = load_model('./src/models/FrameQC_model.keras')


def classificar_imagem(caminho_img):
    img = image.load_img(caminho_img, target_size=(320, 320))
    arr = image.img_to_array(img) / 1.0 / 255
    arr = np.expand_dims(arr, axis=0)    # cria batch de tamanho 1
    prob = model.predict(arr)[0][0]      # probabilidade de "correta"
    return prob
