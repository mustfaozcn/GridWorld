# Reinforcement Learning ile GridWorld

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![NumPy](https://img.shields.io/badge/numpy-1.21+-orange.svg)

Bu proje, GridWorld ortamında farklı reinforcement learning algoritmalarını sıfırdan NumPy ile implement eder ve görselleştirir. Her algoritma, derin öğrenme mimarilerinin reinforcement learning'de nasıl kullanıldığını gösterir.

## Algoritmalar

Proje aşağıdaki algoritmaları içerir:

- **Q-Learning** - Tabular Q-Learning, sinir ağı olmadan Q-tablosu kullanır
- **DQN** - Deep Q-Network, MLP ile Q-değerlerini öğrenir
- **CNN-DQN** - Convolutional neural network ile görüntü formatında öğrenme
- **LSTM-DQN** - Geçmişi hatırlayan LSTM tabanlı ağ
- **Transformer-DQN** - Self-attention mekanizması kullanan ağ
- **Attention-DQN** - Attention mekanizması ile odaklanma
- **ResNet-DQN** - Skip connection'lar ile derin ağ

Her algoritma için:
- `*_numpy.py` - NumPy ile implement edilmiş eğitim kodu
- `*_visualized.py` - Pygame ile görselleştirme içeren kod
- `*.md` - Algoritmanın nasıl çalıştığını açıklayan döküman

## Kurulum

### 1. Sanal Ortam Oluştur

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 3. Çalıştır

Eğitim için:
```bash
python q_learning.py
python deep_q_network_numpy.py
python cnn_numpy.py
# ... diğerleri
```

Görselleştirme ile:
```bash
python deep_q_network_visualized.py
python cnn_visualized.py
# ... diğerleri
```

## Örnek Çıktı

Q-Learning eğitimi sonrası:

```
Öğrenilen Politika (G=Hedef):
→ ↓ ↓ ↓ ↓
→ → ↓ ↓ ↓
→ → → ↓ ↓
→ → → → ↓
→ → → → G

Ortalama değerlendirme ödülü: 2.0
```

## GridWorld Ortamı

5x5 grid üzerinde:
- Ajan başlangıç noktasından (0,0) hedefe (4,4) ulaşmaya çalışır
- Eylemler: Yukarı, Aşağı, Sol, Sağ
- Ödül: Hedefe ulaşınca +10, diğer adımlarda -1

## Dokümantasyon

Her algoritmanın nasıl çalıştığını anlamak için ilgili `.md` dosyalarına bakabilirsiniz:
- `q_learning.md`
- `deep_q_network.md`
- `cnn.md`
- `lstm.md`
- `attention.md`
- `transformer.md`
- `resnet.md`

## Gereksinimler

- Python 3.8+
- NumPy >= 1.21.0
- Pygame >= 2.0.0 (görselleştirme için)

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır. Lütfen pull request gönderirken:
- Kodunuzun mevcut standartlara uygun olduğundan emin olun
- Yeni algoritmalar için ilgili görselleştirme ve dokümantasyon dosyalarını ekleyin
