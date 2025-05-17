import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# 1. Define onde estão as suas duas classes: "certos" e "errados"
base_path = "dataset"
# O flow_from_directory espera subpastas com nomes de classe (aqui, 'certos' e 'errados') :contentReference[oaicite:0]{index=0}
train_path = os.path.join(base_path)  # irá varrer dataset/certos e dataset/errados automaticamente

# 2. Configura o gerador de imagens com augmentações simples
datagen = ImageDataGenerator(
    rescale=1./255,            # normaliza pixels de [0–255] para [0–1] :contentReference[oaicite:1]{index=1}
    validation_split=0.2,      # reserva 20% das imagens para validação :contentReference[oaicite:2]{index=2}
    rotation_range=10,         # gira levemente (±10°)
    brightness_range=[0.8,1.2],# varia o brilho
    zoom_range=0.1             # aplica zoom leve
)

# 3. Cria iteradores de treino e validação

# Para treinar, usa 80% das imagens
train_gen = datagen.flow_from_directory(
    train_path,                # caminho pai contendo 'certos' e 'errados'
    target_size=(128, 128),    # redimensiona todas as imagens para 128×128
    batch_size=16,             # quantas imagens por iteração
    classes=['Errados','Certos'],  # explicitamente define errados→0, certos→1
    class_mode='binary',       # saída 0 ou 1 (errado/certo)
    subset='training'          # pega 80% das imagens para treino
)

# Para testarr, usa 20% das imagens
val_gen = datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=16,
    classes=['Errados','Certos'],  # mesma ordem para consistência
    class_mode='binary',
    subset='validation'        # pega os 20% restantes para validação
)

# 4. Define uma CNN simples
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),              # achata o volume para vetor
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # sigmoide para probabilidade binária
])

# 5. Compila o modelo
model.compile(
    optimizer='adam',              # otimizador padrão eficiente
    loss='binary_crossentropy',    # loss para 2 classes
    metrics=['accuracy']           # mede acurácia durante o treino
)

# 6. Treina e captura histórico
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# 7. Plota a evolução da acurácia
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.legend()
plt.show()

# 8. Salva o modelo para uso posterior
model.save('modelo_classificacao.h5')
print('Modelo salvo em modelo_classificacao.h5')
