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
TARGET_SIZE = (300, 300)  # Reduzir tamanho para economizar memória
INITIAL_BATCH = 6  # Reduzir batch inicial
VALIDATION_SPLIT = 0.2
SEED = 123
EPOCHS = 100  # Aumentar épocas para melhor convergência

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

    # map: resize para TARGET_SIZE e aplicar múltiplos filtros de bordas
    def preprocess(x, y):
        # Redimensiona a imagem para o tamanho alvo
        x = tf.image.resize(x, TARGET_SIZE)
        
        # Normalização melhorada da imagem original
        x_norm = tf.cast(x, tf.float32) / 255.0
        
        # Aplica múltiplos filtros para melhor detecção de características
        sobel = tf.image.sobel_edges(x_norm)
        sobel = tf.reshape(sobel, [-1, TARGET_SIZE[0], TARGET_SIZE[1], 2])
        
        # Normalização melhorada do Sobel - preservar mais informação
        sobel_magnitude = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1, keepdims=True))
        sobel_normalized = tf.where(sobel_magnitude > 0, 
                                  sobel / (sobel_magnitude + 1e-8), 
                                  sobel)
        
        # Adicionar canal de magnitude das bordas
        magnitude_channel = tf.clip_by_value(sobel_magnitude, 0, 1)
        
        # Adicionar filtro Laplaciano para detectar bordas de segunda ordem
        kernel_laplaciano = tf.constant([[[[-1]], [[-1]], [[-1]]],
                                      [[[-1]], [[8]], [[-1]]],
                                      [[[-1]], [[-1]], [[-1]]]], dtype=tf.float32)
        laplacian = tf.nn.conv2d(x_norm, kernel_laplaciano, strides=[1,1,1,1], padding='SAME')
        laplacian = tf.clip_by_value(laplacian, -1, 1) * 0.5 + 0.5
        
        # Combinar todos os canais: Sobel_x, Sobel_y, Magnitude, Laplacian
        combined = tf.concat([sobel_normalized, magnitude_channel, laplacian], axis=-1)
        
        return combined, y

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, val_ds

# tenta com batch inicial, reduz se estourar memória durante dataset loading
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

# === 6) Definição da CNN otimizada para memória ===
input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 4)  # 4 canais: sobel_x, sobel_y, magnitude, laplacian
model = models.Sequential([
    layers.Input(shape=input_shape),
    
    # Bloco 1 - Detecção de características básicas (reduzido)
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.1),
    
    # Bloco 2 - Características intermediárias (reduzido)
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.2),
    
    # Bloco 3 - Características complexas (reduzido)
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.3),
    
    # Bloco 4 - Características de alto nível (reduzido)
    layers.Conv2D(256, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),  # Substitui Flatten + Dense para reduzir overfitting
    
    # Classificador simplificado
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
model.summary()

# === 7) Compile com otimizador melhorado ===
optimizer = optimizers.Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC(name='auc')]
)

# === 8) Callbacks melhorados ===
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models', 'cnn')
os.makedirs(models_dir, exist_ok=True)
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, min_delta=0.001),
    callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.3, patience=5, min_lr=1e-7, verbose=1),
    callbacks.ModelCheckpoint(
        os.path.join(models_dir, 'best_cnn_adapt.keras'),
        monitor='val_auc', mode='max', save_best_only=True, verbose=1
    ),
    callbacks.TensorBoard(log_dir=os.path.join(models_dir, 'logs'))
]

# === 9) Treinamento com tratamento de OOM ===
def train_with_oom_handling():
    global model  # Garantir acesso à variável global
    global callbacks_list  # <-- Adicione esta linha
    current_batch = batch_size
    
    while current_batch >= 1:
        try:
            print(f"Tentando treinar com batch_size={current_batch}")
            
            # Rebuild datasets com novo batch_size se necessário
            if current_batch != batch_size:
                train_ds_new, val_ds_new = build_datasets('dataset', current_batch)
                train_ds_new = train_ds_new.cache().shuffle(buffer_size).prefetch(AUTOTUNE)
                val_ds_new = val_ds_new.cache().prefetch(AUTOTUNE)
            else:
                train_ds_new, val_ds_new = train_ds, val_ds
            
            history = model.fit(
                train_ds_new,
                validation_data=val_ds_new,
                epochs=EPOCHS,
                callbacks=callbacks_list,
                class_weight=class_weights
            )
            return history, train_ds_new, val_ds_new
            
        except tf.errors.ResourceExhaustedError as e:
            print(f"OOM durante treinamento com batch_size={current_batch}")
            current_batch = max(current_batch // 2, 1)
            if current_batch < 1:
                raise RuntimeError("Não foi possível treinar nem com batch_size=1") from e
            
            # Clear GPU memory
            tf.keras.backend.clear_session()
            
            # Rebuild model with same architecture
            model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv2D(32, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(32, 3, activation='relu', padding='same'),
                layers.MaxPooling2D(2),
                layers.Dropout(0.1),
                layers.Conv2D(64, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(64, 3, activation='relu', padding='same'),
                layers.MaxPooling2D(2),
                layers.Dropout(0.2),
                layers.Conv2D(128, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(128, 3, activation='relu', padding='same'),
                layers.MaxPooling2D(2),
                layers.Dropout(0.2),
                layers.Conv2D(256, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=optimizers.Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                loss='binary_crossentropy',
                metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC(name='auc')]
            )
            
            # Recriar callbacks com novos caminhos para evitar conflitos
            callbacks_list_new = [
                callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, min_delta=0.001),
                callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.3, patience=5, min_lr=1e-7, verbose=1),
                callbacks.ModelCheckpoint(
                    os.path.join(models_dir, f'best_cnn_adapt_batch{current_batch}.keras'),
                    monitor='val_auc', mode='max', save_best_only=True, verbose=1
                ),
                callbacks.TensorBoard(log_dir=os.path.join(models_dir, f'logs_batch{current_batch}'))
            ]
            
            # Atualizar callbacks para próxima tentativa
            callbacks_list = callbacks_list_new
            
            print(f"Modelo reconstruído, tentando novamente com batch_size={current_batch}")

# Executar treinamento com tratamento de OOM
history, final_train_ds, final_val_ds = train_with_oom_handling()

# === 10) Plots métricas ===
plt.figure(figsize=(14,5))
for i, m in enumerate(['loss','accuracy','auc']):
    plt.subplot(1,3,i+1)
    plt.plot(history.history[m], label='treino', marker='o')
    plt.plot(history.history[f'val_{m}'], label='val', marker='s')
    plt.title(m.capitalize())
    plt.xlabel('Época')
    plt.ylabel(m.capitalize())
    plt.grid(True, linestyle='--', alpha=0.6)
    # Destacar valores máximos/mínimos
    train_vals = history.history[m]
    val_vals = history.history[f'val_{m}']
    if m == 'loss':
        idx_train = np.argmin(train_vals)
        idx_val = np.argmin(val_vals)
        plt.scatter(idx_train, train_vals[idx_train], color='blue', zorder=5)
        plt.scatter(idx_val, val_vals[idx_val], color='orange', zorder=5)
        plt.annotate(f'{train_vals[idx_train]:.3f}', (idx_train, train_vals[idx_train]), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        plt.annotate(f'{val_vals[idx_val]:.3f}', (idx_val, val_vals[idx_val]), textcoords="offset points", xytext=(0,-15), ha='center', color='orange')
    else:
        idx_train = np.argmax(train_vals)
        idx_val = np.argmax(val_vals)
        plt.scatter(idx_train, train_vals[idx_train], color='blue', zorder=5)
        plt.scatter(idx_val, val_vals[idx_val], color='orange', zorder=5)
        plt.annotate(f'{train_vals[idx_train]:.3f}', (idx_train, train_vals[idx_train]), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        plt.annotate(f'{val_vals[idx_val]:.3f}', (idx_val, val_vals[idx_val]), textcoords="offset points", xytext=(0,-15), ha='center', color='orange')
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(models_dir,'metrics_adapt.png'), dpi=200)
plt.show()

# === 11) Avaliação melhorada com threshold otimizado ===
print("\nAvaliação Final:")
val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(final_val_ds)
print(f"AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")

# Otimizar threshold baseado na validação
y_true, y_pred_proba = [], []
for imgs, labs in final_val_ds:
    preds_proba = model.predict(imgs, verbose=0).flatten()
    y_true.extend(labs.numpy().astype(int))
    y_pred_proba.extend(preds_proba)

y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)

# Encontrar melhor threshold testando vários valores
from sklearn.metrics import f1_score, precision_score, recall_score
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba > thresh).astype(int)
    f1 = f1_score(y_true, y_pred_thresh)
    f1_scores.append(f1)

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"\nMelhor threshold: {best_threshold:.3f}")
print(f"F1-Score com threshold otimizado: {best_f1:.4f}")

# Usar threshold otimizado para matriz de confusão
y_pred_optimized = (y_pred_proba > best_threshold).astype(int)

# Métricas detalhadas
precision_opt = precision_score(y_true, y_pred_optimized)
recall_opt = recall_score(y_true, y_pred_optimized)
print(f"Precision otimizada: {precision_opt:.4f}")
print(f"Recall otimizada: {recall_opt:.4f}")

cm = confusion_matrix(y_true, y_pred_optimized)
plt.figure(figsize=(8,7))
im = plt.imshow(cm, cmap=plt.cm.Blues)
plt.title(f'Matriz de Confusão\n(Threshold otimizado: {best_threshold:.3f})', fontsize=14)
plt.xlabel('Predito', fontsize=12)
plt.ylabel('Real', fontsize=12)

# Calcular métricas da matriz de confusão
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

plt.xticks(np.arange(len(class_counts)), [f'Neg\n(TN={tn}, FP={fp})', f'Pos\n(FN={fn}, TP={tp})'])
plt.yticks(np.arange(len(class_counts)), ['Neg (Real)', 'Pos (Real)'])
plt.colorbar(im, fraction=0.046, pad=0.04)

thresh_color = cm.max()/2
total = cm.sum()
for i,j in np.ndindex(cm.shape):
    pct = cm[i,j]/total*100
    plt.text(j,i, f'{cm[i,j]}\n({pct:.1f}%)', ha='center', va='center',
             color='white' if cm[i,j]>thresh_color else 'black', fontsize=12, fontweight='bold')

# Adicionar métricas no gráfico
textstr = f'Sensibilidade: {sensitivity:.3f}\nEspecificidade: {specificity:.3f}\nF1-Score: {best_f1:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(models_dir,'confusion_adapt.png'), dpi=200)
plt.show()

# Análise de erros por confiança
print(f"\nAnálise de Erros:")
print(f"Falsos Positivos: {fp} ({fp/total*100:.1f}%)")
print(f"Falsos Negativos: {fn} ({fn/total*100:.1f}%)")

# Distribuição de confiança dos erros - CORRIGIDO
# Use np.logical_and() em vez de operador & para evitar problemas de precedência
fp_mask = np.logical_and(y_true == 0, y_pred_optimized == 1)
fn_mask = np.logical_and(y_true == 1, y_pred_optimized == 0)

# Corrigir indexação para garantir que as máscaras sejam 1D
fp_confidence = y_pred_proba[fp_mask.ravel()]
fn_confidence = y_pred_proba[fn_mask.ravel()]

# Verificação de segurança
print(f"Encontrados {len(fp_confidence)} falsos positivos para análise")
print(f"Encontrados {len(fn_confidence)} falsos negativos para análise")

if len(fp_confidence) > 0:
    print(f"Confiança média dos FP: {fp_confidence.mean():.3f} ± {fp_confidence.std():.3f}")
if len(fn_confidence) > 0:
    print(f"Confiança média dos FN: {fn_confidence.mean():.3f} ± {fn_confidence.std():.3f}")

# Histograma de distribuição de confiança dos erros
if len(fp_confidence) > 0 and len(fn_confidence) > 0:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(fp_confidence, bins=20, alpha=0.7, color='red')
    plt.title('Distribuição de Confiança - Falsos Positivos')
    plt.xlabel('Confiança (Probabilidade)')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(fn_confidence, bins=20, alpha=0.7, color='blue')
    plt.title('Distribuição de Confiança - Falsos Negativos')
    plt.xlabel('Confiança (Probabilidade)')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'confianca_erros.png'), dpi=200)
    plt.show()

# === 12) Salvar modelo final ===
model.save(os.path.join(models_dir,'cnn_final_adapt.keras'))
print("\nModelo adaptativo salvo em 'models/'.")