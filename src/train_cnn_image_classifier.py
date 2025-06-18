# Script de treinamento de modelo CNN para classificar imagens "verdadeiras" e "falsas" com fine-tuning e early stopping.

import os
import sys
import subprocess
import pickle
# Verifica e instala matplotlib se necessário
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# 1) Dataset customizado
class CustomImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        img = self.transform(image=img)['image']
        return img, label

# 2) Função principal
def main():
    # Verifica dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Carrega dados estruturados em 'dataset/verdadeiros' e 'dataset/falsos'
    from torchvision.datasets import ImageFolder
    full = ImageFolder('dataset')
    samples = full.samples
    train_idx, val_idx = train_test_split(
        list(range(len(samples))), test_size=0.2,
        stratify=full.targets, random_state=42)

    # 3) Transforms sutis e focados
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=256, p=1.0),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, p=1.0),
        A.CenterCrop(height=224, width=224, p=1.0),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=(3,5), p=0.1),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    # DataLoaders
    train_ds = CustomImageDataset([samples[i] for i in train_idx], train_transform)
    val_ds   = CustomImageDataset([samples[i] for i in val_idx],   val_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    # 4) Modelo com fine-tuning parcial
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for name, param in base.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    num_feats = base.fc.in_features
    base.fc = nn.Linear(num_feats, 1)
    model = base.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 5) Treinamento com early stopping
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    num_epochs = 50
    best_val_loss = float('inf')
    patience, counter = 5, 0

    for epoch in range(num_epochs):
        # Treino
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f'Ép {epoch+1}/{num_epochs}'):  
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*imgs.size(0)
            preds = (torch.sigmoid(outputs)>0.5).float()
            correct += (preds==labels).sum().item()
            total += labels.size(0)
        train_loss, train_acc = running_loss/total, correct/total

        # Validação
        model.eval()
        val_loss_sum, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(imgs)
                vloss = criterion(outputs, labels)
                val_loss_sum += vloss.item()*imgs.size(0)
                preds = (torch.sigmoid(outputs)>0.5).float()
                val_correct += (preds==labels).sum().item()
                val_total += labels.size(0)
        val_loss, val_acc = val_loss_sum/val_total, val_correct/val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'[{epoch+1}/{num_epochs}] loss: {train_loss:.4f}, acc: {train_acc:.4f} –'
              f' val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping após {epoch+1} épocas. Melhor val_loss: {best_val_loss:.4f}")
                break

    # 6) Plotagem e salvamento
    epochs = list(range(1, len(history['train_loss'])+1))
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(epochs, history['train_acc'], label='Treino')
    plt.plot(epochs, history['val_acc'], label='Validação')
    plt.ylabel('Acurácia'); plt.legend(); plt.title('Acurácia')
    plt.subplot(2,1,2)
    plt.plot(epochs, history['train_loss'], label='Treino')
    plt.plot(epochs, history['val_loss'], label='Validação')
    plt.ylabel('Loss'); plt.xlabel('Época'); plt.legend(); plt.title('Loss')
    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.show()

    # 7) Salvar modelo e histórico
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/modelo_cnn.pth')
    with open('models/history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print('Modelo, gráfico e histórico salvos.')

if __name__ == '__main__':
    main()
