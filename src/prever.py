from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Carrega o modelo salvo
model = load_model('FrameQC_model.keras')

def classificar_imagem(caminho_img):
    img = image.load_img(caminho_img, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)    # cria batch de tamanho 1
    prob = model.predict(arr)[0][0]      # probabilidade de "correta"
    return prob


if __name__ == "__main__":
    # Use uma raw string (r) para evitar que Python interprete os caracteres de escape
    prob = classificar_imagem(r'E:\Py_Projetos\FrameQC\dataset\Certos\frame_004140.jpg')
    if prob >= 0.5:
        print(f'Imagem classificada como correta (confiança {prob:.2f})')
    else:
        print(f'Imagem classificada como errada (confiança {prob:.2f})')