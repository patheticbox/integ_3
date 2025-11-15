import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchaudio
import os
import time

# Встановлення пристзю
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Параметри
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# Обмежена кількість класів
CLASSES = ['yes', 'no', 'up', 'down']
NUM_CLASSES = len(CLASSES)

class SpeechCommandsSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.indices = []
        
        # Фільтруємо лише потрібні класи
        for i in range(len(dataset)):
            label = dataset[i][2]
            if label in classes:
                self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        waveform, sample_rate, label, *_ = self.dataset[real_idx]
        label_idx = self.class_to_idx[label]
        return waveform, sample_rate, label_idx

# Модель CNN для класифікації спектрограм
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Розмір після згортки залежить від вхідних даних
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return x

def preprocess_audio(waveform, sample_rate, target_length=16000):
    # Перетворення на моно, якщо стерео
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Обрізка або доповнення до фіксованої довжини
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    # Перетворення у Mel-спектрограму
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    mel_spec = mel_transform(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return mel_spec

def collate_fn(batch):
    waveforms = []
    labels = []
    
    for waveform, sample_rate, label in batch:
        mel_spec = preprocess_audio(waveform, sample_rate)
        waveforms.append(mel_spec)
        labels.append(label)
    
    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels)
    
    return waveforms, labels

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            start_time = time.time()
            output = model(data)
            inference_time = (time.time() - start_time) * 1000  # в мс
            inference_times.append(inference_time)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    avg_latency = sum(inference_times) / len(inference_times)
    
    return avg_loss, accuracy, avg_latency

def main():
    print("Loading dataset Speech Commands...")
    
    # Завантаження датасету
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS('./', download=True, subset='training')
    test_dataset = torchaudio.datasets.SPEECHCOMMANDS('./', download=True, subset='testing')
    
    # Фільтрування класів
    train_subset = SpeechCommandsSubset(train_dataset, CLASSES)
    test_subset = SpeechCommandsSubset(test_dataset, CLASSES)
    
    print(f"Train subjects amount: {len(train_subset)}")
    print(f"Test subjects amount: {len(test_subset)}")
    print(f"Classes: {CLASSES}")
    
    # DataLoader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    
    # Ініціалізація моделі
    model = SimpleCNN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Підрахунок розміру моделі
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal amount of model parameters: {total_params:,}")
    
    # Навчання
    print("\n" + "="*50)
    print("START TRAINING")
    print("="*50)
    
    for epoch in range(EPOCHS):
        print(f"\Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        test_loss, test_acc, avg_latency = evaluate(model, test_loader, criterion, device)
        print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, Latency: {avg_latency:.2f} мс")
    
    # Фінальна оцінка
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    final_loss, final_acc, final_latency = evaluate(model, test_loader, criterion, device)
    print(f"Accuracy: {final_acc:.2f}%")
    print(f"Latency: {final_latency:.2f} мс")
    
    # Збереження моделі
    model_path = 'speech_command_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': CLASSES,
        'n_mels': N_MELS,
        'sample_rate': SAMPLE_RATE
    }, model_path)
    
    # Розмір моделі у файлі
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} МБ")
    
    print(f"\nМodel had been saved to: {model_path}")

if __name__ == '__main__':
    main()