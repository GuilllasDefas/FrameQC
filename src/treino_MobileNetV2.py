# --- Configuração Inicial e Supressão de Avisos ---
import os
import warnings
import logging
import datetime # Para logs do TensorBoard

# Suprimir avisos gerais e do TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_DETERMINISTIC_OPS'] = '1' # Para reprodutibilidade (pode impactar performance)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suprimir mensagens de log do TensorFlow

import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# --- Hiperparâmetros e Configurações Globais ---
# Dataset
DATASET_DIR = 'dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 123 # Seed para reprodutibilidade

# Treinamento
INITIAL_LR = 0.00005 # Ajustado para fine-tuning, pode ser experimentado
DECAY_STEPS = 10000 # Ajustado para um decaimento mais gradual
DECAY_RATE = 0.9
EPOCHS = 30 # Aumentado, EarlyStopping controlará o número real
PATIENCE_EARLY_STOPPING = 7 # Paciência para EarlyStopping
FINE_TUNE_AT_LAYER = 100  # Camada a partir da qual o fine-tuning começa na MobileNetV2 (ajuste conforme necessário)

# Arquitetura do Modelo (após o modelo base)
DENSE_UNITS_1 = 256
DROPOUT_1 = 0.5
DENSE_UNITS_2 = 128
DROPOUT_2 = 0.3

# --- Configuração de Seeds para Reprodutibilidade ---
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Preparação dos Diretórios ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__ if '__file__' in globals() else '.')) # Robustez para diferentes ambientes
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
LOGS_DIR = os.path.join(SCRIPT_DIR, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Carregamento e Preparação do Dataset ---
print("\n--- Carregando Datasets ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary', # Para classificação binária (0 ou 1)
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Salvar os nomes das classes
CLASS_NAMES = train_ds.class_names
print(f"Classes encontradas: {CLASS_NAMES}")
print(f"Total de amostras de treino: {tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE}")
print(f"Total de amostras de validação: {tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE}")

# --- Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
    layers.RandomContrast(0.1, seed=SEED),
], name='data_augmentation')

# --- Otimização de Performance do Dataset ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Cálculo de Pesos de Classe (para lidar com desbalanceamento) ---
print("\n--- Calculando Pesos de Classe ---")
labels_list = []
for _, labels_batch in train_ds:
    labels_list.append(labels_batch.numpy())
labels_array = np.concatenate(labels_list, axis=0).astype(int).flatten()

if labels_array.size > 0:
    class_counts = np.bincount(labels_array)
    if len(class_counts) == len(CLASS_NAMES): # Garante que temos contagens para todas as classes
        total_samples = np.sum(class_counts)
        class_weights = {i: total_samples / (count * len(CLASS_NAMES)) for i, count in enumerate(class_counts) if count > 0}
        print(f"Distribuição de classes no treino (antes do peso): {dict(enumerate(class_counts))}")
        print(f"Pesos de classes aplicados: {class_weights}")
    else:
        print("Aviso: Número de classes contadas não corresponde ao esperado. Não aplicando pesos de classe.")
        class_weights = None
else:
    print("Aviso: Não foi possível extrair labels para cálculo de pesos. Não aplicando pesos de classe.")
    class_weights = None


# --- Construção do Modelo com Fine-Tuning ---
print("\n--- Construindo Modelo ---")
# Camada de reescala para normalizar as imagens para o range [-1, 1] esperado pela MobileNetV2
rescale_layer = layers.Rescaling(1./127.5, offset=-1, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Modelo base (MobileNetV2)
base_model = applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False, # Não incluir a camada classificadora original
    weights='imagenet'
)

# Congelar o modelo base inicialmente
base_model.trainable = False

# Permitir que as camadas a partir de FINE_TUNE_AT_LAYER sejam treinadas
for layer in base_model.layers[FINE_TUNE_AT_LAYER:]:
    layer.trainable = True
print(f"Modelo base: {len(base_model.layers)} camadas. Fine-tuning a partir da camada {FINE_TUNE_AT_LAYER}.")
num_trainable_layers_base = sum(1 for layer in base_model.layers if layer.trainable)
print(f"Número de camadas treináveis no modelo base: {num_trainable_layers_base}")


# Adicionar camadas personalizadas no topo
model = models.Sequential([
    rescale_layer,
    data_augmentation, # Aplicar augmentation como parte do modelo
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(), # Normaliza ativações antes da camada Dense
    layers.Dense(DENSE_UNITS_1, activation='relu'),
    layers.Dropout(DROPOUT_1),
    layers.Dense(DENSE_UNITS_2, activation='relu'),
    layers.Dropout(DROPOUT_2),
    layers.Dense(1, activation='sigmoid') # Camada de saída para classificação binária
], name='Custom_MobileNetV2')

# --- Compilação do Modelo ---
# Usar um learning rate schedule para ajustar a taxa de aprendizado durante o treino
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LR,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=True # Opcional: LR muda em intervalos discretos
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)
model.summary()

# --- Callbacks ---
print("\n--- Configurando Callbacks ---")
# EarlyStopping para interromper o treino se não houver melhora
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE_EARLY_STOPPING,
    verbose=1,
    restore_best_weights=True # Restaura os pesos do modelo da época com o melhor valor da métrica monitorada
)

# ModelCheckpoint para salvar o melhor modelo com base na acurácia de validação
model_checkpoint_accuracy = ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR, 'best_model_val_accuracy.keras'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# TensorBoard para visualização
tensorboard_callback = TensorBoard(
    log_dir=LOGS_DIR,
    histogram_freq=1, # Calcular histogramas de ativação e pesos (pode consumir recursos)
    write_graph=True
)

# NOTA: ReduceLROnPlateau NÃO é usado aqui porque já estamos usando um LearningRateSchedule (ExponentialDecay)
# Usar ambos para o mesmo otimizador causaria o TypeError que você observou.

callbacks_list = [early_stopping, model_checkpoint_accuracy, tensorboard_callback]

# --- Treinamento do Modelo ---
print("\n--- Iniciando Treinamento ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    class_weight=class_weights if class_weights else None # Aplicar pesos se calculados
)

# --- Plotar Curvas de Treinamento ---
print("\n--- Gerando Gráficos de Treinamento ---")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.title('Perda (Loss) Durante Treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.plot(history.history.get('precision', []), label='Precisão Treino') # .get para caso a métrica não seja registrada
plt.plot(history.history.get('val_precision', []), label='Precisão Validação')
plt.plot(history.history.get('recall', []), label='Recall Treino')
plt.plot(history.history.get('val_recall', []), label='Recall Validação')
plt.title('Métricas Durante Treinamento')
plt.xlabel('Época')
plt.ylabel('Valor da Métrica')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))
plt.show()

# --- Avaliação Final ---
# O modelo 'model' já possui os pesos restaurados por EarlyStopping (melhor val_loss)
print("\n--- Avaliação do Modelo com Melhor 'val_loss' (Restaurado por EarlyStopping) ---")
val_loss, val_acc, val_prec, val_rec = model.evaluate(val_ds, verbose=1)
print(f"Perda Validação: {val_loss:.4f}")
print(f"Acurácia Validação: {val_acc:.4f}")
print(f"Precisão Validação: {val_prec:.4f}")
print(f"Recall Validação: {val_rec:.4f}")
if (val_prec + val_rec) > 0:
    f1_score = 2 * (val_prec * val_rec) / (val_prec + val_rec)
    print(f"F1-Score Validação: {f1_score:.4f}")
else:
    print("F1-Score Validação: 0.0000 (divisão por zero evitada)")

# Salvar este modelo (melhor val_loss)
model.save(os.path.join(MODELS_DIR, 'FrameQC_model.keras'))
print(f"Modelo com melhor 'val_loss' salvo em: {os.path.join(MODELS_DIR, 'FrameQC_model.keras')}")

# Carregar e avaliar o modelo com melhor 'val_accuracy' (salvo por ModelCheckpoint)
best_accuracy_model_path = os.path.join(MODELS_DIR, 'best_model_val_accuracy.keras')
if os.path.exists(best_accuracy_model_path):
    print("\n--- Avaliação do Modelo com Melhor 'val_accuracy' (De ModelCheckpoint) ---")
    model_best_acc = tf.keras.models.load_model(best_accuracy_model_path)
    val_loss_ba, val_acc_ba, val_prec_ba, val_rec_ba = model_best_acc.evaluate(val_ds, verbose=1)
    print(f"Perda Validação (Melhor Acurácia): {val_loss_ba:.4f}")
    print(f"Acurácia Validação (Melhor Acurácia): {val_acc_ba:.4f}")
    print(f"Precisão Validação (Melhor Acurácia): {val_prec_ba:.4f}")
    print(f"Recall Validação (Melhor Acurácia): {val_rec_ba:.4f}")
    if (val_prec_ba + val_rec_ba) > 0:
        f1_score_ba = 2 * (val_prec_ba * val_rec_ba) / (val_prec_ba + val_rec_ba)
        print(f"F1-Score Validação (Melhor Acurácia): {f1_score_ba:.4f}")
    else:
        print("F1-Score Validação (Melhor Acurácia): 0.0000 (divisão por zero evitada)")

    # --- Matriz de Confusão e Relatório de Classificação (para o modelo de melhor acurácia) ---
    print("\n--- Gerando Matriz de Confusão para Modelo de Melhor Acurácia ---")
    y_true_list = []
    y_pred_probs_list = []
    for images_batch, labels_batch in val_ds:
        predictions_batch = model_best_acc.predict_on_batch(images_batch) # Usar predict_on_batch
        y_pred_probs_list.extend(predictions_batch.flatten())
        y_true_list.extend(labels_batch.numpy().astype(int).flatten())

    y_pred_classes = [1 if prob > 0.5 else 0 for prob in y_pred_probs_list]

    # Função para plotar matriz de confusão (movida para cá para ser usada)
    def plot_confusion_matrix(cm, class_names_list, title='Matriz de Confusão', save_path=None):
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names_list))
        plt.xticks(tick_marks, class_names_list, rotation=45)
        plt.yticks(tick_marks, class_names_list)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Prevista')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    cm = confusion_matrix(y_true_list, y_pred_classes)
    plot_confusion_matrix(cm, CLASS_NAMES,
                          title='Matriz de Confusão (Modelo Melhor Acurácia)',
                          save_path=os.path.join(MODELS_DIR, 'confusion_matrix_best_acc.png'))

    print("\nRelatório de Classificação (Modelo Melhor Acurácia):")
    print(classification_report(y_true_list, y_pred_classes, target_names=CLASS_NAMES))
else:
    print(f"Modelo {best_accuracy_model_path} não encontrado. Pulando avaliação e matriz de confusão.")

print("\n--- Treinamento e Avaliação Concluídos ---")