import os
import warnings
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# === 1) Supressão de logs TF ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# === 2) Parâmetros ===
ORIG_HEIGHT = 1080  # ajuste para altura original
ORIG_WIDTH  = 1920  # ajuste para largura original
TARGET_SIZE = (360, 360)  # tamanho seguro para a rede
INITIAL_BATCH = 32
VALIDATION_SPLIT = 0.2
SEED = 123
EPOCHS = 50

# === 3) Construir datasets com dynamic resizing e pre-filter (edges) ===
def build_datasets(data_dir, batch_size):
    # carrega original em grayscale para treino
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred', label_mode='binary',
        validation_split=VALIDATION_SPLIT,
        subset='training', seed=SEED,
        image_size=(ORIG_HEIGHT, ORIG_WIDTH), color_mode='grayscale',
        batch_size=batch_size
    )
    
    # carrega original em grayscale para validação
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred', label_mode='binary',
        validation_split=VALIDATION_SPLIT,
        subset='validation', seed=SEED,
        image_size=(ORIG_HEIGHT, ORIG_WIDTH), color_mode='grayscale',
        batch_size=batch_size
    )

    # map: resize para TARGET_SIZE e aplicar sobel para destacar bordas
    def preprocess(x, y):
        # Redimensiona a imagem para o tamanho alvo
        x = tf.image.resize(x, TARGET_SIZE)
        # Aplica o filtro Sobel para destacar bordas
        sobel = tf.image.sobel_edges(x)
        # Reformata diretamente para ter 2 canais (gx, gy)
        # sobel tem formato [batch, height, width, channels=1, sobel_channels=2]
        # queremos [batch, height, width, sobel_channels=2]
        sobel = tf.reshape(sobel, [-1, TARGET_SIZE[0], TARGET_SIZE[1], 2])
        # Normaliza os valores
        sobel = sobel / tf.reduce_max(tf.abs(sobel) + 1e-9) * 0.5 + 0.5
        return sobel, y

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, val_ds

# tenta com batch inicial, reduz se estourar memória
batch_size = INITIAL_BATCH
while True:
    try:
        train_ds, val_ds = build_datasets('dataset', batch_size)
        break
    except tf.errors.ResourceExhaustedError:
        batch_size = max(batch_size // 2, 1)
        print(f"OOM ao carregar batch_size={batch_size*2}, tentando {batch_size}...")

print(f"Usando batch_size={batch_size}")

# === 4) Ajustar pipeline tf.data ===
# Não contar elementos no dataset, isso consome o iterador
# Use o tamanho fornecido pelo dataset
buffer_size = 1000  # Valor fixo para o buffer de shuffle

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(buffer_size).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# === 5) Pesos de classes ===
labels = np.concatenate([y for x, y in train_ds], axis=0)
class_counts = np.bincount(labels.astype(int).flatten())
total = labels.shape[0]
class_weights = {i: total/count for i, count in enumerate(class_counts)}
print("Distribuição de classes:", class_counts)
print("Pesos de classes:", class_weights)

# === 6) Definição da CNN própria ===
input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 2)  # sobel gera 2 canais (gx, gy)
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, 3, activation='relu', padding='same'), layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu', padding='same'), layers.MaxPooling2D(2),
    layers.Conv2D(128, 3, activation='relu', padding='same'), layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'), layers.Dropout(0.3),
    layers.Dense(128, activation='relu'), layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
model.summary()

# === 7) Compile com adaptador de LR e ReduceLROnPlateau ===
optimizer = optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC(name='auc')]
)

# === 8) Callbacks para adaptar LR e monitorar métricas ===
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models', 'cnn')
os.makedirs(models_dir, exist_ok=True)
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, min_lr=1e-6),
    callbacks.ModelCheckpoint(
        os.path.join(models_dir, 'best_cnn_adapt.keras'),
        monitor='val_auc', mode='max', save_best_only=True
    ),
    callbacks.TensorBoard(log_dir=os.path.join(models_dir, 'logs'))
]

# === 9) Treinamento com adaptação automática de batch e LR ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    class_weight=class_weights
)

# === 10) Plots métricas ===
plt.figure(figsize=(12,5))
for i, m in enumerate(['loss','accuracy','auc']):
    plt.subplot(1,3,i+1)
    plt.plot(history.history[m], label='treino')
    plt.plot(history.history[f'val_{m}'], label='val')
    plt.title(m.capitalize())
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(models_dir,'metrics_adapt.png'))
plt.show()

# === 11) Avaliação e matriz de confusão ===
print("\nAvaliação Final:")
val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_ds)
print(f"AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")

y_true, y_pred = [], []
for imgs, labs in val_ds:
    preds = (model.predict(imgs) > 0.5).astype(int).flatten()
    y_true.extend(labs.numpy().astype(int))
    y_pred.extend(preds)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap=plt.cm.Greys)
plt.title('Matriz de Confusão')
plt.xticks(np.arange(len(class_counts)), ['Neg','Pos'], rotation=45)
plt.yticks(np.arange(len(class_counts)), ['Neg','Pos'])
thresh=cm.max()/2
for i,j in np.ndindex(cm.shape): plt.text(j,i,cm[i,j],ha='center', color='white' if cm[i,j]>thresh else 'black')
plt.tight_layout()
plt.savefig(os.path.join(models_dir,'confusion_adapt.png'))
plt.show()

# === 12) Salvar modelo final ===
model.save(os.path.join(models_dir,'cnn_final_adapt.keras'))
print("\nModelo adaptativo salvo em 'models/'.")