from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch import nn
from torchvision import models

# Ayarlar
UPLOAD_FOLDER = 'Uploaded_Files'
IM_SIZE = 112
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Basit model sarmalayıcı (senin orijinal yapınla uyumlu olacak şekilde)
class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        base = models.resnext50_32x4d(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Görüntü dönüşümleri
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Dataset: video -> sabit sayıda frame
class ValidationDataset(Dataset):
    def __init__(self, video_paths, sequence_length=60, transform=None):
        self.video_paths = video_paths
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = []
        cap = cv2.VideoCapture(video_path)
        success = True
        while success and len(frames) < self.count:
            success, frame = cap.read()
            if not success:
                break
            if frame is None:
                continue
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IM_SIZE, IM_SIZE))
            if self.transform:
                frames.append(self.transform(frame))
            else:
                frames.append(torch.from_numpy(frame).permute(2,0,1).float()/255.0)
        cap.release()
        if len(frames) == 0:
            raise ValueError('Video karesi bulunamadı veya okunamadı.')
        # stack -> (sequence, C, H, W)
        frames = torch.stack(frames)
        return frames

# Tahmin fonksiyonu (UI için hafif ve güvenilir)
def predict(model, frames_tensor):
    # frames_tensor: (sequence, C, H, W)
    model.eval()
    preds = []
    soft = nn.Softmax(dim=1)
    with torch.no_grad():
        for i in range(frames_tensor.shape[0]):
            x = frames_tensor[i].unsqueeze(0).to(DEVICE)
            logits = model(x)
            probs = soft(logits).cpu().numpy()[0]  # [p_fake, p_real]
            p_fake = float(probs[0])
            p_real = float(probs[1])
            label = 'FAKE' if p_fake > p_real else 'REAL'
            preds.append({'frame': i, 'label': label, 'p_fake': p_fake, 'p_real': p_real})
    return preds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Detect', methods=['POST'])
def detect():
    if 'video' not in request.files:
        return jsonify({'error': 'video dosyası bulunamadı.'}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'dosya adı boş.'}), 400

    filename = secure_filename(video.filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(save_path)

    try:
        # Dönüşümleri hazırla
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        dataset = ValidationDataset([save_path], sequence_length=20, transform=transform)
        frames = dataset[0]  # (sequence, C, H, W)

        # Modeli yükle
        model_path = os.path.join('model', 'df_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError('model/df_model.pt bulunamadı. Model dosyasını model/ dizinine koyun.')

        model = Model(num_classes=2).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        # Eğer kaydedilen state dict doğrudan state_dict ise kullan, değilse ayarla
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'], strict=False)
        else:
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                # Bazı modeller doğrudan torch.save(model) olarak kaydedilmiş olabilir
                model = state
        model.to(DEVICE)

        preds = predict(model, frames)

        # Basit istatistikler
        fake_scores = [p['p_fake'] for p in preds]
        real_scores = [p['p_real'] for p in preds]
        avg_fake = float(np.mean(fake_scores)) if fake_scores else 0.0
        avg_real = float(np.mean(real_scores)) if real_scores else 0.0
        final_label = 'FAKE' if avg_fake > avg_real else 'REAL'
        confidence = max(avg_fake, avg_real)

        result = {
            'result': final_label,
            'confidence': round(confidence * 100, 2),
            'per_frame': preds
        }

    except Exception as e:
        print('Hata:', e)
        result = {'error': str(e)}

    finally:
        # temizle
        if os.path.exists(save_path):
            os.remove(save_path)

    return jsonify(result)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    print('Uygulama başlatılıyor — Cihaz:', DEVICE)
    app.run(host='0.0.0.0', port=5000, debug=True)