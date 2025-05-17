from tensorflow.keras.preprocessing import image
import numpy as np

def classificar_imagem(caminho_img):
    img = image.load_img(caminho_img, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)    # cria batch de tamanho 1
    prob = model.predict(arr)[0][0]      # probabilidade de “correta”
    return 'Correta' if prob >= 0.5 else 'Incorreta', prob

# teste rápido
rotulo, prob = classificar_imagem('dataset/teste/exemplo.jpg')
print(f'Imagem classificada como {rotulo} (confiança {prob:.2f})')