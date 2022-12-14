
# -*- coding: utf-8 -*-
"""
NumPy ile Sıfırdan Convolutional Neural Network (CNN) ile Deep Q-Learning (GridWorld)
--------------------------------------------------------------------------------------
- Bu dosya, GridWorld problemini CNN (Convolutional Neural Network) kullanarak çözer.
- MLP yerine, GridWorld'i 2D görüntü olarak temsil edip convolutional katmanlar
  kullanarak uzamsal desenleri öğrenir.
- CNN'in Gücü: Lokal komşuluk ilişkilerini ve uzamsal desenleri yakalayabilir.
  Örneğin, ajanın hedefe göre konumunu, duvarlara yakınlığını, geometrik desenleri
  otomatik olarak öğrenir.
- Anahtar Konseptler:
    1. Görüntü Temsili: GridWorld'i çok kanallı (3 kanal) görüntü olarak kodlama.
    2. Convolution: Küçük filtreler (kernels) ile lokal özellikleri çıkarma.
    3. MaxPooling: Özellik haritalarını örnekleyerek boyutu küçültme.
    4. Flatten ve Dense: Özellikleri Q-değerlerine dönüştürme.
Bu dosyayı tek başına çalıştırabilirsiniz.
"""
# ---------------------------------------------------------------------
# OKUMA SIRASI TAVSİYESİ
# ---------------------------------------------------------------------
# 1. `GridWorld` Sınıfı: Ajanın içinde yaşadığı ortam.
# 2. `to_image_grid` Fonksiyonu: Durumu görüntüye dönüştürme (CNN için kritik).
# 3. `if __name__ == "__main__"` Bloğu: Kodun ana akışı.
# 4. `train_cnn_dqn` Fonksiyonu: CNN-DQN eğitim sürecini yöneten ana döngü.
# 5. `CNN` Sınıfı: Convolutional sinir ağı (beyin).
# 6. `ReplayBuffer` Sınıfı: Ajanın deneyimlerini sakladığı hafıza.
# 7. Diğer yardımcı fonksiyonlar (`epsilon_greedy` vb.).
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, List
import numpy as np
import random

# Ortak modüllerden import
from common import GridWorld, epsilon_greedy

# ---------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------------------
def to_image_grid(s: Tuple[int, int], w: int, h: int, goal: Tuple[int, int]) -> np.ndarray:
    """
    Durumu (x,y) 3 kanallı görüntü olarak temsil eder.
    
    CNN'ler görüntüler üzerinde çalışır, bu yüzden GridWorld'i görüntüye
    dönüştürmeliyiz. Bu fonksiyon, her hücre için 3 bilgiyi 3 ayrı kanalda
    kodlar:
    
    Kanal 0 (Ajan Kanalı): Ajanın bulunduğu hücre 1.0, diğerleri 0.0
    Kanal 1 (Hedef Kanalı): Hedefin bulunduğu hücre 1.0, diğerleri 0.0
    Kanal 2 (Mesafe Kanalı): Her hücreden hedefe olan normalizasyonlu mesafe
    
    Bu temsil sayesinde CNN, uzamsal ilişkileri (ajan hedefin neresinde,
    mesafe ne kadar, vs.) otomatik olarak öğrenebilir.
    
    Örnek: 5x5 grid için (2,1) konumunda ajan, (4,4) hedef
    Çıktı: (5, 5, 3) boyutunda bir görüntü
    
    Args:
        s: Durum koordinatları (x, y)
        w: GridWorld genişliği
        h: GridWorld yüksekliği
        goal: Hedef koordinatları (x, y)
    
    Returns:
        3 kanallı görüntü (h, w, 3)
    
    Raises:
        ValueError: Geçersiz parametre değerleri
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"Grid dimensions must be positive, got w={w}, h={h}")
    if not isinstance(s, (tuple, list)) or len(s) != 2:
        raise ValueError(f"State must be a tuple/list of 2 integers, got {s}")
    if not isinstance(goal, (tuple, list)) or len(goal) != 2:
        raise ValueError(f"Goal must be a tuple/list of 2 integers, got {goal}")
    
    x, y = s
    goal_x, goal_y = goal
    
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"State ({x}, {y}) out of bounds for grid size {w}x{h}")
    if not (0 <= goal_x < w and 0 <= goal_y < h):
        raise ValueError(f"Goal ({goal_x}, {goal_y}) out of bounds for grid size {w}x{h}")
    
    # 3 kanallı görüntü oluştur: (yükseklik, genişlik, kanal)
    img = np.zeros((h, w, 3), dtype=np.float32)
    
    # Kanal 0: Ajan pozisyonu (one-hot encoding)
    img[y, x, 0] = 1.0
    
    # Kanal 1: Hedef pozisyonu (one-hot encoding)
    img[goal_y, goal_x, 1] = 1.0
    
    # Kanal 2: Her hücreden hedefe olan normalizasyonlu Öklid mesafesi
    # Bu, ağın "yön hissi" kazanmasına yardımcı olur.
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - goal_y)**2 + (j - goal_x)**2)
            max_dist = np.sqrt((h-1)**2 + (w-1)**2)  # En uzak mesafe
            img[i, j, 2] = dist / (max_dist + 1e-6)  # Normalize et (0-1 arası)
    
    return img

# epsilon_greedy artık common modülünden import ediliyor

# ---------------------------------------------------------------------
# 4. "BEYİN": CONVOLUTIONAL NEURAL NETWORK (CNN)
# Q-değerlerini tahmin etmek için kullanılan convolutional sinir ağı.
# CNN'in avantajı, görüntüdeki lokal desenleri ve uzamsal ilişkileri
# otomatik olarak öğrenebilmesidir. MLP'den farklı olarak, her nöron
# tüm görüntüye değil, sadece küçük bir bölgeye (receptive field) bakar.
# Bu sayede ağ, ajanın hedefe göre konumunu, duvarlara yakınlığını,
# geometrik desenleri daha verimli öğrenir.
# ---------------------------------------------------------------------
class CNN:
    """
    Basit bir Convolutional Neural Network.
    Mimari: Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → Dense → Q-değerleri
    """
    def __init__(
        self, 
        img_h: int = 5, 
        img_w: int = 5, 
        img_c: int = 3, 
        conv1_filters: int = 16, 
        conv2_filters: int = 32,
        dense_hidden: int = 64, 
        out_dim: int = 4, 
        lr: float = 1e-3, 
        seed: int = 42
    ) -> None:
        """
        CNN'in başlatılması.
        
        Args:
            img_h, img_w, img_c: Girdi görüntüsünün boyutları (yükseklik, genişlik, kanal)
            conv1_filters: İlk convolutional katmandaki filtre sayısı
            conv2_filters: İkinci convolutional katmandaki filtre sayısı
            dense_hidden: Flatten sonrası dense katmanın nöron sayısı
            out_dim: Çıktı boyutu (Q-değerleri için 4 eylem)
            lr: Öğrenme oranı
            seed: Rastgele sayı üreteci için tohum
        
        Raises:
            ValueError: Geçersiz parametre değerleri
        """
        if img_h <= 0 or img_w <= 0 or img_c <= 0:
            raise ValueError(f"Image dimensions must be positive, got img_h={img_h}, img_w={img_w}, img_c={img_c}")
        if conv1_filters <= 0 or conv2_filters <= 0 or dense_hidden <= 0 or out_dim <= 0:
            raise ValueError(f"All filter/hidden dimensions must be positive, got conv1_filters={conv1_filters}, conv2_filters={conv2_filters}, dense_hidden={dense_hidden}, out_dim={out_dim}")
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        rng = np.random.default_rng(seed)
        
        # --- 1. İLK CONVOLUTIONAL KATMAN (Conv1) ---
        # Conv2D: 3x3 filtreler, padding='same' (görüntü boyutu korunur)
        # Her filtre, görüntü üzerinde kayarak lokal özellikleri çıkarır.
        # Filtre boyutu: (filtre_yüksekliği, filtre_genişliği, girdi_kanalları, çıktı_kanalları)
        self.conv1_filters = 3  # 3x3 filtre boyutu
        self.conv1_W = rng.standard_normal((3, 3, img_c, conv1_filters)).astype(np.float32) * np.sqrt(2.0 / (3*3*img_c))
        self.conv1_b = np.zeros((conv1_filters,), dtype=np.float32)
        self.conv1_filters_count = conv1_filters
        
        # MaxPooling: 2x2 pencereler, stride=2 (görüntü boyutu yarıya iner)
        # MaxPooling parametre yok, sadece operasyon
        
        # --- 2. İKİNCİ CONVOLUTIONAL KATMAN (Conv2) ---
        # İlk pool sonrası görüntü boyutu: (img_h//2, img_w//2)
        self.pooled_h = img_h // 2
        self.pooled_w = img_w // 2
        
        self.conv2_filters = 3  # 3x3 filtre boyutu
        self.conv2_W = rng.standard_normal((3, 3, conv1_filters, conv2_filters)).astype(np.float32) * np.sqrt(2.0 / (3*3*conv1_filters))
        self.conv2_b = np.zeros((conv2_filters,), dtype=np.float32)
        self.conv2_filters_count = conv2_filters
        
        # İkinci pool sonrası görüntü boyutu: (pooled_h//2, pooled_w//2)
        self.final_h = self.pooled_h // 2
        self.final_w = self.pooled_w // 2
        
        # --- 3. FLATTEN ve DENSE KATMANLAR ---
        # Flatten: Tüm feature map'leri tek bir vektöre dönüştür
        flatten_size = self.final_h * self.final_w * conv2_filters
        
        # Dense katman (gizli)
        self.dense_W = rng.standard_normal((flatten_size, dense_hidden)).astype(np.float32) * np.sqrt(2.0 / flatten_size)
        self.dense_b = np.zeros((dense_hidden,), dtype=np.float32)
        
        # Çıkış katmanı (Q-değerleri)
        self.out_W = rng.standard_normal((dense_hidden, out_dim)).astype(np.float32) * np.sqrt(2.0 / dense_hidden)
        self.out_b = np.zeros((out_dim,), dtype=np.float32)
        
        self.lr = lr
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ ReLU aktivasyon fonksiyonu. """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        """ ReLU'nun türevi (geri yayılım için). """
        return (x > 0).astype(np.float32)
    
    @staticmethod
    def conv2d_forward(X, W, b, stride=1, padding='same'):
        """
        2D Convolution işlemi (forward pass).
        
        Convolution, bir görüntü üzerinde küçük filtrelerin (kernels) kayarak
        geçmesi ve her pozisyonda lokal bir özellik çıkarmasıdır. Örneğin,
        bir filtre "dikey çizgileri" arayabilir, başka bir filtre "köşeleri"
        arayabilir. Bu sayede CNN, görüntüdeki kompleks desenleri öğrenir.
        
        Parametreler:
        - X: Girdi görüntüsü (batch, height, width, channels)
        - W: Filtre ağırlıkları (filter_h, filter_w, in_channels, out_channels)
        - b: Sapma değerleri (out_channels,)
        - stride: Filtrenin kaç piksel kaydığı
        - padding: 'same' = görüntü boyutunu koru, 'valid' = boyutu küçült
        
        Çıktı: (batch, out_height, out_width, out_channels)
        """
        batch, in_h, in_w, in_c = X.shape
        f_h, f_w, _, out_c = W.shape
        
        if padding == 'same':
            # Padding ekle: görüntüyü sarmalayarak (wraparound) veya sıfırlarla doldurarak
            # Kenarları da filtre uygulanabilir hale getiriyoruz.
            pad_h = (f_h - 1) // 2
            pad_w = (f_w - 1) // 2
            X_padded = np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:  # valid
            X_padded = X
            pad_h = pad_w = 0
        
        # Çıktı boyutlarını hesapla
        out_h = (in_h + 2 * pad_h - f_h) // stride + 1
        out_w = (in_w + 2 * pad_w - f_w) // stride + 1
        
        # Çıktı dizisini başlat
        out = np.zeros((batch, out_h, out_w, out_c), dtype=np.float32)
        
        # Her filtre için convolution işlemi
        for i in range(out_h):
            for j in range(out_w):
                # Filtrenin görüntüdeki konumu
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + f_h
                w_end = w_start + f_w
                
                # Bu bölgeyi çıkar ve filtre ile çarp (element-wise)
                X_slice = X_padded[:, h_start:h_end, w_start:w_end, :]
                
                # Matris çarpımı: (batch, f_h, f_w, in_c) @ (f_h, f_w, in_c, out_c)
                # Her filtre için ayrı ayrı hesapla
                for oc in range(out_c):
                    out[:, i, j, oc] = np.sum(X_slice * W[:, :, :, oc], axis=(1, 2, 3)) + b[oc]
        
        return out
    
    @staticmethod
    def maxpool2d_forward(X, pool_size=2, stride=2):
        """
        MaxPooling işlemi (forward pass).
        
        MaxPooling, bir görüntüyü örnekleyerek (downsampling) boyutunu küçültür.
        Her pool_size x pool_size bölgeden en büyük değeri alır. Bu işlem:
        1. Görüntüyü küçültür (hesaplama maliyetini düşürür)
        2. Özellikleri daha genel hale getirir (örn: "sol üstte bir şey var" gibi)
        3. Overfitting'i azaltır (daha az parametre)
        
        Parametreler:
        - X: Girdi feature map (batch, height, width, channels)
        - pool_size: Pool penceresi boyutu (örn: 2x2)
        - stride: Kayma miktarı (genelde pool_size ile aynı)
        
        Çıktı: (batch, out_height, out_width, channels)
        """
        batch, in_h, in_w, channels = X.shape
        
        out_h = in_h // stride
        out_w = in_w // stride
        
        out = np.zeros((batch, out_h, out_w, channels), dtype=np.float32)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + pool_size
                w_end = w_start + pool_size
                
                # Bu bölgedeki maksimum değeri al
                X_slice = X[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.max(X_slice, axis=(1, 2))
        
        return out
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """
        İleri yayılım (Forward Pass): Girdi görüntüsünü alır ve Q-değerlerini hesaplar.
        
        Akış:
        1. Conv1 → ReLU → MaxPool: İlk seviye özellikler (kenarlar, köşeler)
        2. Conv2 → ReLU → MaxPool: İkinci seviye özellikler (karmaşık desenler)
        3. Flatten: Özellik haritalarını vektöre dönüştür
        4. Dense → ReLU: Özellikleri birleştir
        5. Output: Q-değerleri
        
        Args:
            X: Girdi görüntüleri (batch, height, width, channels)
        
        Returns:
            Tuple containing:
                - q: Q-değerleri (batch, out_dim)
                - cache: Geri yayılım için ara değerler
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if X.ndim != 4:
            raise ValueError(f"Input must be 4D (batch, height, width, channels), got shape {X.shape}")
        # Conv1: İlk seviye özellik çıkarma
        conv1_out = self.conv2d_forward(X, self.conv1_W, self.conv1_b)
        conv1_act = self.relu(conv1_out)  # Aktivasyon
        
        # MaxPool1: Örnekleme
        pool1_out = self.maxpool2d_forward(conv1_act)
        
        # Conv2: İkinci seviye özellik çıkarma
        conv2_out = self.conv2d_forward(pool1_out, self.conv2_W, self.conv2_b)
        conv2_act = self.relu(conv2_out)  # Aktivasyon
        
        # MaxPool2: Örnekleme
        pool2_out = self.maxpool2d_forward(conv2_act)
        
        # Flatten: (batch, h, w, c) -> (batch, h*w*c)
        batch = pool2_out.shape[0]
        flattened = pool2_out.reshape(batch, -1)
        
        # Dense (gizli katman)
        dense_out = flattened @ self.dense_W + self.dense_b
        dense_act = self.relu(dense_out)
        
        # Çıkış katmanı (Q-değerleri)
        q = dense_act @ self.out_W + self.out_b
        
        # Cache: Geri yayılım için ara değerleri sakla
        cache = (X, conv1_out, conv1_act, pool1_out, 
                 conv2_out, conv2_act, pool2_out, flattened, 
                 dense_out, dense_act)
        
        return q, cache
    
    def backward(self, cache: Tuple, dq: np.ndarray) -> None:
        """
        Geri yayılım (Backward Pass): Hata gradyanını kullanarak ağırlıkları günceller.
        
        Args:
            cache: Forward pass'ten gelen ara değerler
            dq: Çıktı katmanı gradyanları (batch, out_dim)
        
        Raises:
            ValueError: Gradyan boyutu beklenen boyutla eşleşmiyorsa
        """
        if dq.shape[1] != self.out_W.shape[1]:
            raise ValueError(f"Gradient shape mismatch: expected (batch, {self.out_W.shape[1]}), got {dq.shape}")
        
        Convolution ve MaxPooling'in geri yayılımı özel işlemler gerektirir:
        - Conv geri yayılımı: Filtre ağırlıklarını günceller
        - MaxPool geri yayılımı: Sadece maksimum değerlere gradyan gönderir
        - Dense geri yayılımı: Standart matris çarpımı gradyanları
        """
        (X, conv1_out, conv1_act, pool1_out,
         conv2_out, conv2_act, pool2_out, flattened,
         dense_out, dense_act) = cache
        
        batch = X.shape[0]
        
        # --- ÇIKIŞ KATMANI GRADYANLARI ---
        # dq: (batch, out_dim)
        d_dense_act = dq @ self.out_W.T  # (batch, dense_hidden)
        d_out_W = dense_act.T @ dq / batch  # (dense_hidden, out_dim)
        d_out_b = dq.mean(axis=0)  # (out_dim,)
        
        # --- DENSE KATMAN GRADYANLARI ---
        d_dense_out = d_dense_act * self.relu_grad(dense_out)  # (batch, dense_hidden)
        d_flattened = d_dense_out @ self.dense_W.T  # (batch, flatten_size)
        d_dense_W = flattened.T @ d_dense_out / batch  # (flatten_size, dense_hidden)
        d_dense_b = d_dense_out.mean(axis=0)  # (dense_hidden,)
        
        # --- FLATTEN GERİ ALMA ---
        # Flatten'ın geri yayılımı: Vektörü görüntü şekline geri çevir
        pool2_shape = pool2_out.shape  # (batch, h, w, c)
        d_pool2_out = d_flattened.reshape(pool2_shape)
        
        # --- MAXPOOL2 GERİ YAYILIMI ---
        # MaxPool geri yayılımı: Sadece maksimum değerlere gradyan gönder
        d_conv2_act = self.maxpool2d_backward(d_pool2_out, conv2_act, pool_size=2, stride=2)
        
        # --- CONV2 GRADYANLARI ---
        d_conv2_out = d_conv2_act * self.relu_grad(conv2_out)
        d_conv2_W, d_conv2_b, d_pool1_out = self.conv2d_backward(pool1_out, self.conv2_W, d_conv2_out)
        
        # --- MAXPOOL1 GERİ YAYILIMI ---
        d_conv1_act = self.maxpool2d_backward(d_pool1_out, conv1_act, pool_size=2, stride=2)
        
        # --- CONV1 GRADYANLARI ---
        d_conv1_out = d_conv1_act * self.relu_grad(conv1_out)
        d_conv1_W, d_conv1_b, _ = self.conv2d_backward(X, self.conv1_W, d_conv1_out)
        
        # Ağırlıkları güncelle
        self.out_W -= self.lr * d_out_W
        self.out_b -= self.lr * d_out_b
        self.dense_W -= self.lr * d_dense_W
        self.dense_b -= self.lr * d_dense_b
        self.conv2_W -= self.lr * d_conv2_W
        self.conv2_b -= self.lr * d_conv2_b
        self.conv1_W -= self.lr * d_conv1_W
        self.conv1_b -= self.lr * d_conv1_b
    
    @staticmethod
    def conv2d_backward(X, W, d_out):
        """
        Convolution geri yayılımı.
        Filtre ağırlıklarının ve girdi görüntüsünün gradyanlarını hesaplar.
        
        Bu fonksiyon, zincir kuralını kullanarak convolution işleminin
        geri yayılımını yapar. Her filtre için, girdi görüntüsünün ilgili
        bölgesi ile çıktı gradyanını çarparak filtre ağırlıklarının
        gradyanını bulur.
        """
        batch, in_h, in_w, in_c = X.shape
        f_h, f_w, _, out_c = W.shape
        _, out_h, out_w, _ = d_out.shape
        
        # Padding ekle (forward pass ile aynı)
        pad_h = (f_h - 1) // 2
        pad_w = (f_w - 1) // 2
        X_padded = np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        
        # Gradyanları başlat
        d_W = np.zeros_like(W)
        d_b = np.zeros((out_c,), dtype=np.float32)
        d_X_padded = np.zeros_like(X_padded)
        
        # Her çıktı pozisyonu için
        for i in range(out_h):
            for j in range(out_w):
                h_start = i
                w_start = j
                h_end = h_start + f_h
                w_end = w_start + f_w
                
                X_slice = X_padded[:, h_start:h_end, w_start:w_end, :]
                
                # Her filtre için
                for oc in range(out_c):
                    # Filtre ağırlık gradyanı: X_slice ile d_out'u çarp
                    d_W[:, :, :, oc] += np.mean(X_slice * d_out[:, i:i+1, j:j+1, oc:oc+1], axis=0)
                    
                    # Girdi gradyanı: Filtre ağırlıkları ile d_out'u çarp
                    d_X_padded[:, h_start:h_end, w_start:w_end, :] += (
                        W[:, :, :, oc:oc+1] * d_out[:, i:i+1, j:j+1, oc:oc+1]
                    )
        
        # Bias gradyanı
        d_b = d_out.sum(axis=(0, 1, 2)) / batch
        
        # Padding'i geri al
        if pad_h > 0 or pad_w > 0:
            d_X = d_X_padded[:, pad_h:-pad_h if pad_h > 0 else None, 
                                   pad_w:-pad_w if pad_w > 0 else None, :]
        else:
            d_X = d_X_padded
        
        # Filtre gradyanlarını normalize et
        d_W /= batch
        
        return d_W, d_b, d_X
    
    @staticmethod
    def maxpool2d_backward(d_out, X, pool_size=2, stride=2):
        """
        MaxPooling geri yayılımı.
        Sadece maksimum değerlere gradyan gönderir, diğerlerine 0.
        
        MaxPooling'in geri yayılımı, forward pass'te hangi değerlerin
        maksimum olduğunu hatırlamamızı gerektirir. Bu fonksiyon, her
        pool bölgesinde maksimum değerin nerede olduğunu bulup, sadece
        o pozisyona gradyan gönderir.
        """
        batch, out_h, out_w, channels = d_out.shape
        d_X = np.zeros_like(X)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + pool_size
                w_end = w_start + pool_size
                
                X_slice = X[:, h_start:h_end, w_start:w_end, :]
                
                # Her batch ve kanal için maksimum değerin yerini bul
                for b in range(batch):
                    for c in range(channels):
                        pool_region = X_slice[b, :, :, c]
                        max_val = np.max(pool_region)
                        max_positions = (pool_region == max_val)
                        
                        # Maksimum değerlere gradyan gönder
                        d_X[b, h_start:h_end, w_start:w_end, c] += (
                            max_positions * d_out[b, i, j, c] / np.sum(max_positions)
                        )
        
        return d_X
    
    def predict(self, x_single: np.ndarray) -> np.ndarray:
        """ 
        Tek bir görüntü için Q-değerlerini tahmin eder.
        
        Args:
            x_single: Tek görüntü (height, width, channels)
        
        Returns:
            Q-değerleri (out_dim,)
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if x_single.ndim != 3:
            raise ValueError(f"Input must be 3D (height, width, channels), got shape {x_single.shape}")
        q, _ = self.forward(x_single[None, :, :, :])
        return q[0]
    
    def copy_from(self, other: 'CNN') -> None:
        """ 
        Başka bir CNN'in ağırlıklarını bu ağa kopyalar (Target Network için).
        
        Args:
            other: Kopyalanacak CNN nesnesi
        
        Raises:
            ValueError: Ağ yapıları uyumsuzsa
        """
        if (self.conv1_W.shape != other.conv1_W.shape or 
            self.conv2_W.shape != other.conv2_W.shape or 
            self.dense_W.shape != other.dense_W.shape or 
            self.out_W.shape != other.out_W.shape):
            raise ValueError("Network architectures must match")
        
        self.conv1_W = other.conv1_W.copy()
        self.conv1_b = other.conv1_b.copy()
        self.conv2_W = other.conv2_W.copy()
        self.conv2_b = other.conv2_b.copy()
        self.dense_W = other.dense_W.copy()
        self.dense_b = other.dense_b.copy()
        self.out_W = other.out_W.copy()
        self.out_b = other.out_b.copy()

# ---------------------------------------------------------------------
# 5. "HAFIZA": DENEYİM TEKRARI TAMPONU (REPLAY BUFFER)
# Ajanın (durum, eylem, ödül, yeni_durum) deneyimlerini saklar.
# Bu sefer durumlar görüntü formatında (5x5x3).
# ---------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        """
        ReplayBuffer'ı başlatır.
        
        Args:
            capacity: Buffer kapasitesi (pozitif tam sayı)
        
        Raises:
            ValueError: Kapasite pozitif değilse
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.buf: List[Tuple] = []
        self.pos = 0
    
    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: float) -> None:
        """ 
        Bir deneyimi hafızaya ekler.
        
        Args:
            s: Mevcut durum görüntüsü
            a: Yapılan eylem
            r: Alınan ödül
            s2: Yeni durum görüntüsü
            d: Bölüm bitme durumu (0.0 veya 1.0)
        """
        data = (s, a, r, s2, d)
        if len(self.buf) < self.capacity:
            self.buf.append(data)
        else:
            self.buf[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Hafızadan rastgele bir batch deneyim çeker.
        
        Args:
            batch: Örnek boyutu (pozitif tam sayı)
        
        Returns:
            Tuple containing (S, A, R, S2, D) numpy arrays
        
        Raises:
            ValueError: Batch boyutu geçersizse veya buffer yeterince dolu değilse
        """
        if batch <= 0:
            raise ValueError(f"Batch size must be positive, got {batch}")
        if len(self.buf) < batch:
            raise ValueError(f"Not enough samples in buffer: {len(self.buf)} < {batch}")
        
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        S, A, R, S2, D = zip(*[self.buf[i] for i in idx])
        return (
            np.stack(S), 
            np.array(A), 
            np.array(R, dtype=np.float32), 
            np.stack(S2), 
            np.array(D, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buf)

# ---------------------------------------------------------------------
# GÖRSELLEŞTİRME VE DEĞERLENDİRME
# ---------------------------------------------------------------------
ACTIONS = {0:"↑", 1:"↓", 2:"←", 3:"→"}

def render_policy(env: GridWorld, net: CNN) -> None:
    """ 
    Öğrenilen politikayı ekrana çizer.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş CNN ağı
    """
    grid: List[List[str]] = []
    for y in range(env.h):
        row: List[str] = []
        for x in range(env.w):
            if (x, y) == env.goal:
                row.append("G")
                continue
            img = to_image_grid((x, y), env.w, env.h, env.goal)
            a = int(np.argmax(net.predict(img)))
            if a not in ACTIONS:
                raise ValueError(f"Invalid action {a} found in policy at position ({x}, {y})")
            row.append(ACTIONS[a])
        grid.append(row)
    print("\n(CNN-DQN) Öğrenilen Politika:")
    for r in grid:
        print(" ".join(r))

def evaluate(
    env: GridWorld, 
    net: CNN, 
    episodes: int = 5, 
    max_steps: int = 100
) -> List[float]:
    """ 
    Eğitilmiş ajanın performansını test eder.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş CNN ağı
        episodes: Test bölüm sayısı (pozitif tam sayı)
        max_steps: Her bölüm için maksimum adım sayısı (pozitif tam sayı)
    
    Returns:
        Her bölüm için toplam ödül listesi
    
    Raises:
        ValueError: Geçersiz parametre değerleri
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    
    out: List[float] = []
    for _ in range(episodes):
        s = env.reset()
        R = 0.0
        for _ in range(max_steps):
            img = to_image_grid(s, env.w, env.h, env.goal)
            a = int(np.argmax(net.predict(img)))
            s, r, done, _ = env.step(a)
            R += r
            if done:
                break
        out.append(R)
    return out

# ---------------------------------------------------------------------
# 3. ANA EĞİTİM FONKSİYONU
# Tüm CNN-DQN öğrenme sürecini yönetir.
# ---------------------------------------------------------------------
def train_cnn_dqn(
    episodes: int = 900,
    max_steps: int = 200,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_min: float = 0.01,
    eps_decay: float = 0.995,
    lr: float = 1e-3,
    batch: int = 64,
    buf_cap: int = 5000,
    start_after: int = 400,
    target_every: int = 200,
    use_target: bool = True,
    seed: int = 123
) -> Tuple[GridWorld, CNN, CNN, List[float]]:
    """
    CNN-DQN eğitim fonksiyonu.
    MLP-DQN ile aynı mantık, sadece durum temsili görüntü formatında.
    
    Args:
        episodes: Toplam eğitim bölümü sayısı (pozitif tam sayı)
        max_steps: Her bölüm için maksimum adım sayısı (pozitif tam sayı)
        gamma: İndirgeme faktörü (0.0 ile 1.0 arası)
        eps_start: Başlangıç epsilon değeri (0.0 ile 1.0 arası)
        eps_min: Minimum epsilon değeri (0.0 ile eps_start arası)
        eps_decay: Epsilon azalma oranı (0.0 ile 1.0 arası)
        lr: Öğrenme oranı (pozitif)
        batch: Batch boyutu (pozitif tam sayı)
        buf_cap: Replay buffer kapasitesi (pozitif tam sayı)
        start_after: Eğitime başlamak için gereken minimum örnek sayısı (pozitif tam sayı)
        target_every: Hedef ağ güncelleme sıklığı (pozitif tam sayı)
        use_target: Hedef ağ kullanılsın mı?
        seed: Rastgelelik tohumu
    
    Returns:
        Tuple containing (env, policy, target, returns)
    
    Raises:
        ValueError: Geçersiz hiperparametre değerleri
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be between 0.0 and 1.0, got {gamma}")
    if not 0.0 <= eps_start <= 1.0:
        raise ValueError(f"eps_start must be between 0.0 and 1.0, got {eps_start}")
    if not 0.0 <= eps_min <= eps_start:
        raise ValueError(f"eps_min must be between 0.0 and eps_start ({eps_start}), got {eps_min}")
    if not 0.0 <= eps_decay <= 1.0:
        raise ValueError(f"eps_decay must be between 0.0 and 1.0, got {eps_decay}")
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if buf_cap <= 0:
        raise ValueError(f"buf_cap must be positive, got {buf_cap}")
    if start_after <= 0:
        raise ValueError(f"start_after must be positive, got {start_after}")
    if target_every <= 0:
        raise ValueError(f"target_every must be positive, got {target_every}")
    if start_after > buf_cap:
        raise ValueError(f"start_after ({start_after}) cannot be greater than buf_cap ({buf_cap})")
    
    random.seed(seed)
    np.random.seed(seed)
    env = GridWorld()
    
    # CNN ağları (görüntü formatı için)
    policy = CNN(img_h=env.h, img_w=env.w, img_c=3, 
                 conv1_filters=16, conv2_filters=32,
                 dense_hidden=64, out_dim=4, lr=lr, seed=seed)
    target = CNN(img_h=env.h, img_w=env.w, img_c=3,
                 conv1_filters=16, conv2_filters=32,
                 dense_hidden=64, out_dim=4, lr=lr, seed=seed+1)
    target.copy_from(policy)
    
    rb = ReplayBuffer(buf_cap)
    eps = eps_start
    returns = []
    steps = 0
    
    # Ana eğitim döngüsü
    for ep in range(episodes):
        s = env.reset()
        epR = 0.0
        for t in range(max_steps):
            # Durumu görüntü formatına çevir
            img = to_image_grid(s, env.w, env.h, env.goal)
            
            # Eylem seçimi
            q = policy.predict(img)
            a = epsilon_greedy(q, eps)
            
            # Eylemi gerçekleştir
            s2, r, done, _ = env.step(a)
            img2 = to_image_grid(s2, env.w, env.h, env.goal)
            
            # Deneyimi hafızaya kaydet (görüntü formatında)
            rb.push(img, a, r, img2, float(done))
            epR += r
            s = s2
            steps += 1
            
            # Ağı eğit
            if len(rb) >= max(batch, start_after):
                S, A, R, S2, D = rb.sample(batch)
                
                # Q-değerlerini hesapla
                Qs, cache = policy.forward(S)
                
                if use_target:
                    Qs2, _ = target.forward(S2)
                else:
                    Qs2, _ = policy.forward(S2)
                
                # TD hedefi
                y = R + gamma * (1.0 - D) * np.max(Qs2, axis=1)
                
                # Gradyan hesapla
                dq = np.zeros_like(Qs)
                idx = np.arange(batch)
                dq[idx, A] = (Qs[idx, A] - y)
                
                # Geri yayılım
                policy.backward(cache, dq)
            
            if done: break
            
            # Hedef ağı güncelle
            if use_target and steps % target_every == 0:
                target.copy_from(policy)
        
        returns.append(epR)
        eps = max(eps_min, eps * eps_decay)
    
    return env, policy, target, returns

# ---------------------------------------------------------------------
# 2. ANA AKIŞ (ENTRY POINT)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # CNN-DQN eğitimi
    env, policy, target, returns = train_cnn_dqn()
    # Öğrenilen politikayı göster
    render_policy(env, policy)
    # Performans değerlendirmesi
    print("Değerlendirme (sadece en iyi eylemlerle):", evaluate(env, policy, 10))

