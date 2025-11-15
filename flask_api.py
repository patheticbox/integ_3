from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchaudio
import io
import time
import numpy as np

app = Flask(__name__)

# Параметри (повинні збігатися з навчанням)
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# Глобальні змінні
model = None
device = None
classes = None

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

def load_model():
    global model, device, classes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Завантаження моделі
    checkpoint = torch.load('speech_command_model.pth', map_location=device)
    classes = checkpoint['classes']
    
    model = SimpleCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model is loaded. Classes: {classes}")

def preprocess_audio(waveform, sample_rate, target_length=16000):
    # Перетворення на моно
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Обрізка або доповнення
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    # Mel-спектрограма
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    mel_spec = mel_transform(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return mel_spec

def predict_audio(waveform, sample_rate):
    start_time = time.time()
    
    # Попередня обробка
    mel_spec = preprocess_audio(waveform, sample_rate)
    mel_spec = mel_spec.unsqueeze(0).to(device)  # Додати batch dimension
    
    # Інференс
    with torch.no_grad():
        output = model(mel_spec)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    inference_time = (time.time() - start_time) * 1000  # в мс
    
    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()
    
    # Всі ймовірності
    all_probs = {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
    
    return {
        'predicted_class': predicted_class,
        'confidence': float(confidence_score),
        'latency_ms': float(inference_time),
        'all_probabilities': all_probs
    }

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Перевірка наявності файлу
        if 'audio' not in request.files:
            return jsonify({'error': 'Audio file not found'}), 400
        
        audio_file = request.files['audio']
        
        # Завантаження аудіо
        audio_bytes = audio_file.read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), format="wav")
        
        # Ресемплінг, якщо потрібно
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = SAMPLE_RATE
        
        # Прогноз
        result = predict_audio(waveform, sample_rate)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_text', methods=['POST'])
def predict_text():
    """
    Endpoint for testing text functions (for demo purposes)
    """
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
    
    try:
        data = request.get_json()
        text_command = data.get('command', '').lower()
        
        if text_command not in classes:
            return jsonify({
                'error': f'Command "{text_command}" is not supported',
                'supported_commands': classes
            }), 400
        
        # Для текстової команди просто повертаємо результат
        # В реальному додатку тут був би синтез мовлення або інша логіка
        return jsonify({
            'input_command': text_command,
            'supported': True,
            'message': f'Command "{text_command}" has been recognized',
            'all_classes': classes
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    if classes is None:
        return jsonify({'error': 'Model has not been loaded'}), 500
    
    return jsonify({'classes': classes})

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("DONE")
    print("\nAvailable endpoints:")
    print("  GET  /health - status check")
    print("  GET  /classes - class list")
    print("  POST /predict - audio file prediction")
    print("  POST /predict_text - text command prediction")
    print("\nServer had been started on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False)