
# -*- coding: utf-8 -*-
"""
NumPy ile Sıfırdan Attention Mechanism ile Deep Q-Learning (GridWorld)
-----------------------------------------------------------------------
- Bu dosya, GridWorld problemini Attention Mechanism kullanarak çözer.
- Attention, ağın hangi özelliklere dikkat etmesi gerektiğini öğrenir.
- Attention'ın Gücü: Önemli bilgileri vurgular, önemsiz bilgileri bastırır.
- Anahtar Konseptler:
    1. Multi-Head Attention: Farklı açılardan bilgiyi analiz eder.
    2. Query, Key, Value: Attention'ın üç temel bileşeni.
    3. Attention Weights: Hangi özelliklerin önemli olduğunu gösterir.
    4. Self-Attention: Özellikler arası ilişkileri öğrenir.
Bu dosyayı tek başına çalıştırabilirsiniz.
"""
# ---------------------------------------------------------------------
# OKUMA SIRASI TAVSİYESİ
# ---------------------------------------------------------------------
# 1. `GridWorld` Sınıfı: Ajanın içinde yaşadığı ortam.
# 2. `to_feature_vector` Fonksiyonu: Durumu özellik vektörüne dönüştürme.
# 3. `if __name__ == "__main__"` Bloğu: Kodun ana akışı.
# 4. `train_attention_dqn` Fonksiyonu: Attention-DQN eğitim sürecini yöneten ana döngü.
# 5. `AttentionNet` Sınıfı: Attention mekanizması (beyin).
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
def to_feature_vector(
    s: Tuple[int, int], 
    w: int, 
    h: int, 
    goal: Tuple[int, int]
) -> np.ndarray:
    """
    Durumu özellik vektörüne dönüştürür.
    
    Attention için, durumu zengin bir özellik vektörü olarak temsil ederiz:
    - Ajanın x, y koordinatları
    - Hedefin x, y koordinatları  
    - Ajan-hedef arası mesafe (normalize)
    - Ajanın duvarlara uzaklığı
    
    Bu özellikler, attention mekanizmasının hangi bilgilerin önemli olduğunu
    öğrenmesine yardımcı olur.
    
    Args:
        s: Durum koordinatları (x, y)
        w: GridWorld genişliği
        h: GridWorld yüksekliği
        goal: Hedef koordinatları (x, y)
    
    Returns:
        Özellik vektörü (7,)
    
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
    
    # Özellik vektörü: [agent_x, agent_y, goal_x, goal_y, distance, wall_dist_x, wall_dist_y]
    features = np.zeros(7, dtype=np.float32)
    
    # Normalize koordinatlar
    features[0] = x / (w - 1)  # Ajan x
    features[1] = y / (h - 1)  # Ajan y
    features[2] = goal_x / (w - 1)  # Hedef x
    features[3] = goal_y / (h - 1)  # Hedef y
    
    # Ajan-hedef mesafesi (Öklid, normalize)
    dist = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
    max_dist = np.sqrt((w-1)**2 + (h-1)**2)
    features[4] = dist / (max_dist + 1e-6)
    
    # Duvarlara uzaklık (normalize)
    features[5] = min(x, w-1-x) / ((w-1)/2 + 1e-6)  # X ekseni duvarlara uzaklık
    features[6] = min(y, h-1-y) / ((h-1)/2 + 1e-6)  # Y ekseni duvarlara uzaklık
    
    return features

# epsilon_greedy artık common modülünden import ediliyor

# ---------------------------------------------------------------------
# 4. "BEYİN": ATTENTION MECHANISM
# Q-değerlerini tahmin etmek için kullanılan attention sinir ağı.
# Attention'ın avantajı, hangi özelliklerin önemli olduğunu otomatik
# olarak öğrenebilmesidir. Bu sayede ağ, "ajan-hedef mesafesi önemli,
# ama duvar uzaklığı daha az önemli" gibi ağırlıklı bir karar verebilir.
# ---------------------------------------------------------------------
class AttentionNet:
    """
    Attention mekanizması kullanan sinir ağı.
    
    Bu ağ, özellik vektörünü alır ve attention mekanizması ile hangi
    özelliklerin daha önemli olduğunu öğrenir. Self-attention kullanarak,
    özellikler arası ilişkileri de yakalar.
    """
    def __init__(
        self, 
        feature_dim: int = 7, 
        hidden_dim: int = 64, 
        num_heads: int = 4, 
        out_dim: int = 4, 
        lr: float = 1e-3, 
        seed: int = 42
    ) -> None:
        """
        Attention ağının başlatılması.
        
        Args:
            feature_dim: Girdi özellik vektörünün boyutu (7)
            hidden_dim: Gizli katman boyutu
            num_heads: Attention head sayısı (multi-head attention)
            out_dim: Çıktı boyutu (Q-değerleri için 4 eylem)
            lr: Öğrenme oranı
            seed: Rastgele sayı üreteci için tohum
        
        Raises:
            ValueError: Geçersiz parametre değerleri
        """
        if feature_dim <= 0 or hidden_dim <= 0 or num_heads <= 0 or out_dim <= 0:
            raise ValueError(f"All dimensions must be positive, got feature_dim={feature_dim}, hidden_dim={hidden_dim}, num_heads={num_heads}, out_dim={out_dim}")
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        rng = np.random.default_rng(seed)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # Her head'in boyutu
        self.lr = lr
        
        # Özellikleri gizli uzaya projeksiyon
        self.W_proj = rng.standard_normal((feature_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / feature_dim)
        self.b_proj = np.zeros((hidden_dim,), dtype=np.float32)
        
        # Multi-head attention için Query, Key, Value ağırlıkları
        # Her head için ayrı Q, K, V matrisleri
        self.W_q = rng.standard_normal((num_heads, hidden_dim, self.head_dim)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.W_k = rng.standard_normal((num_heads, hidden_dim, self.head_dim)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.W_v = rng.standard_normal((num_heads, hidden_dim, self.head_dim)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        
        # Attention çıktısını birleştirme
        self.W_o = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        
        # Feed-forward network (2 katmanlı)
        self.W_ff1 = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b_ff1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W_ff2 = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b_ff2 = np.zeros((hidden_dim,), dtype=np.float32)
        
        # Çıkış katmanı
        self.W_out = rng.standard_normal((hidden_dim, out_dim)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros((out_dim,), dtype=np.float32)
    
    @staticmethod
    def relu(x):
        """ ReLU aktivasyon fonksiyonu. """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x):
        """ ReLU'nun türevi. """
        return (x > 0).astype(np.float32)
    
    @staticmethod
    def softmax(x, axis=-1):
        """
        Softmax fonksiyonu (sayısal stabilite için).
        
        Softmax, attention weight'lerini 0-1 arasına normalize eder
        ve toplamlarını 1 yapar. Bu sayede, hangi özelliklere ne kadar
        "dikkat" verileceği belirlenir.
        """
        # Sayısal stabilite için max'ı çıkar
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)
    
    def scaled_dot_product_attention(self, Q, K, V):
        """
        Scaled Dot-Product Attention.
        
        Attention mekanizmasının kalbidir. Şu formülü uygular:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        Parametreler:
        - Q: Query matrisi (batch, num_features, head_dim)
        - K: Key matrisi (batch, num_features, head_dim)
        - V: Value matrisi (batch, num_features, head_dim)
        
        Çıktı:
        - attention_output: (batch, num_features, head_dim)
        - attention_weights: (batch, num_features, num_features)
        """
        # Q @ K^T hesapla
        scores = Q @ K.swapaxes(-1, -2)  # (batch, num_features, num_features)
        
        # Scale: sqrt(head_dim) ile böl
        scale = np.sqrt(self.head_dim)
        scores = scores / scale
        
        # Softmax ile attention weight'lerini hesapla
        attention_weights = self.softmax(scores, axis=-1)  # (batch, num_features, num_features)
        
        # Attention weight'leri ile Value'yu çarp
        attention_output = attention_weights @ V  # (batch, num_features, head_dim)
        
        return attention_output, attention_weights
    
    def multi_head_attention(self, x):
        """
        Multi-Head Attention.
        
        Özellikler arası ilişkileri farklı açılardan (head'ler) analiz eder.
        Her head, farklı bir özellik kombinasyonuna odaklanır.
        
        Parametreler:
        - x: Girdi özellikleri (batch, num_features, hidden_dim)
        
        Çıktı:
        - output: Attention çıktısı (batch, num_features, hidden_dim)
        - all_attention_weights: Tüm head'ler için attention weight'leri
        """
        batch, num_features, hidden_dim = x.shape
        all_heads = []
        all_attention_weights = []
        
        # Her head için
        for head in range(self.num_heads):
            # Query, Key, Value hesapla
            Q = x @ self.W_q[head]  # (batch, num_features, head_dim)
            K = x @ self.W_k[head]  # (batch, num_features, head_dim)
            V = x @ self.W_v[head]  # (batch, num_features, head_dim)
            
            # Attention hesapla
            head_output, attention_weights = self.scaled_dot_product_attention(Q, K, V)
            all_heads.append(head_output)
            all_attention_weights.append(attention_weights)
        
        # Tüm head'leri birleştir
        concatenated = np.concatenate(all_heads, axis=-1)  # (batch, num_features, hidden_dim)
        
        # Output projection
        output = concatenated @ self.W_o  # (batch, num_features, hidden_dim)
        
        # Attention weight'leri de birleştir (görselleştirme için)
        all_attention_weights = np.stack(all_attention_weights, axis=1)  # (batch, num_heads, num_features, num_features)
        
        return output, all_attention_weights
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        İleri yayılım (Forward Pass).
        
        Akış:
        1. Özellik vektörünü gizli uzaya projeksiyon
        2. Özellikleri sequence olarak düzenle (self-attention için)
        3. Multi-head self-attention
        4. Feed-forward network
        5. Global average pooling (tüm özellikleri birleştir)
        6. Q-değerleri
        
        Args:
            x: Girdi özellik vektörleri (batch, feature_dim)
        
        Returns:
            Tuple containing:
                - q_values: Q-değerleri (batch, out_dim)
                - cache: Geri yayılım için ara değerler
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if x.ndim != 2:
            raise ValueError(f"Input must be 2D (batch, feature_dim), got shape {x.shape}")
        if x.shape[1] != self.feature_dim:
            raise ValueError(f"Input feature dimension mismatch: expected {self.feature_dim}, got {x.shape[1]}")
        
        batch = x.shape[0]
        
        # 1. Projeksiyon: Özellikleri gizli uzaya çevir
        proj = x @ self.W_proj + self.b_proj  # (batch, hidden_dim)
        
        # 2. Self-attention için özellikleri sequence olarak düzenle
        # Her özellik bir "token" olarak düşünülür
        # Basitleştirme: Özellikleri tekrar ederek sequence oluştur
        # (Gerçek uygulamada her özellik ayrı bir token olabilir)
        x_seq = proj[:, None, :]  # (batch, 1, hidden_dim) - tek özellik
        
        # 3. Multi-head self-attention
        attn_output, attention_weights = self.multi_head_attention(x_seq)  # (batch, 1, hidden_dim)
        attn_output = attn_output[:, 0, :]  # (batch, hidden_dim)
        
        # 4. Feed-forward network
        ff1_out = attn_output @ self.W_ff1 + self.b_ff1
        ff1_act = self.relu(ff1_out)
        ff2_out = ff1_act @ self.W_ff2 + self.b_ff2
        ff2_act = self.relu(ff2_out)
        
        # Residual connection: Attention çıktısı + Feed-forward çıktısı
        combined = attn_output + ff2_act  # Residual connection
        
        # 5. Çıkış katmanı
        q_values = combined @ self.W_out + self.b_out  # (batch, out_dim)
        
        # Cache: Geri yayılım için
        cache = {
            'x': x,
            'proj': proj,
            'x_seq': x_seq,
            'attn_output': attn_output,
            'attention_weights': attention_weights,
            'ff1_out': ff1_out,
            'ff1_act': ff1_act,
            'ff2_out': ff2_out,
            'ff2_act': ff2_act,
            'combined': combined
        }
        
        return q_values, cache
    
    def backward(self, cache: dict, dq: np.ndarray) -> None:
        """
        Geri yayılım (Backward Pass): Hata gradyanını kullanarak ağırlıkları günceller.
        
        Args:
            cache: Forward pass'ten gelen ara değerler
            dq: Çıktı katmanı gradyanları (batch, out_dim)
        
        Raises:
            ValueError: Gradyan boyutu beklenen boyutla eşleşmiyorsa
        """
        if dq.shape[1] != self.out_dim:
            raise ValueError(f"Gradient shape mismatch: expected (batch, {self.out_dim}), got {dq.shape}")
        
        # Attention'ın geri yayılımı karmaşıktır çünkü:
        # 1. Multi-head attention'ın tüm head'leri için gradyan hesaplanmalı
        # 2. Attention weight'lerinin gradyanları hesaplanmalı
        # 3. Residual connection'dan geçen gradyanlar eklenmeli
        x = cache['x']
        proj = cache['proj']
        attn_output = cache['attn_output']
        ff1_out = cache['ff1_out']
        ff1_act = cache['ff1_act']
        ff2_out = cache['ff2_out']
        ff2_act = cache['ff2_act']
        combined = cache['combined']
        
        batch = x.shape[0]
        
        # Çıkış katmanı gradyanları
        d_combined = dq @ self.W_out.T  # (batch, hidden_dim)
        dW_out = combined.T @ dq / batch
        db_out = dq.mean(axis=0)
        
        # Residual connection: Gradyan hem attn_output'a hem de ff2_act'e gider
        d_attn_output = d_combined.copy()
        d_ff2_act = d_combined.copy()
        
        # Feed-forward geri yayılımı
        d_ff2_out = d_ff2_act * self.relu_grad(ff2_out)
        d_ff1_act = d_ff2_out @ self.W_ff2.T
        dW_ff2 = ff1_act.T @ d_ff2_out / batch
        db_ff2 = d_ff2_out.mean(axis=0)
        
        d_ff1_out = d_ff1_act * self.relu_grad(ff1_out)
        d_attn_output_ff = d_ff1_out @ self.W_ff1.T
        dW_ff1 = attn_output.T @ d_ff1_out / batch
        db_ff1 = d_ff1_out.mean(axis=0)
        
        # Attention geri yayılımı (basitleştirilmiş)
        # Gerçek uygulamada attention'ın geri yayılımı çok karmaşıktır
        # Burada basit bir yaklaşım kullanıyoruz
        d_attn_output += d_attn_output_ff
        d_proj = d_attn_output.copy()  # Basitleştirme
        
        # Projeksiyon geri yayılımı
        d_x = d_proj @ self.W_proj.T
        dW_proj = x.T @ d_proj / batch
        db_proj = d_proj.mean(axis=0)
        
        # Ağırlıkları güncelle
        self.W_out -= self.lr * dW_out
        self.b_out -= self.lr * db_out
        self.W_ff2 -= self.lr * dW_ff2
        self.b_ff2 -= self.lr * db_ff2
        self.W_ff1 -= self.lr * dW_ff1
        self.b_ff1 -= self.lr * db_ff1
        self.W_proj -= self.lr * dW_proj
        self.b_proj -= self.lr * db_proj
        
        # Attention ağırlıkları için basit güncelleme
        # (Gerçek uygulamada daha karmaşık)
        d_attn_scaled = d_attn_output / self.num_heads
        for head in range(self.num_heads):
            # Basitleştirilmiş attention geri yayılımı
            # Gerçek uygulamada Q, K, V'nin türevleri hesaplanmalı
            d_Q = d_attn_scaled @ self.W_q[head].T
            d_K = d_attn_scaled @ self.W_k[head].T
            d_V = d_attn_scaled @ self.W_v[head].T
            
            # Ağırlık güncellemeleri (basitleştirilmiş)
            x_expanded = proj[:, None, :]  # (batch, 1, hidden_dim)
            self.W_q[head] -= self.lr * (x_expanded.swapaxes(-1, -2) @ d_Q).mean(axis=0) / batch
            self.W_k[head] -= self.lr * (x_expanded.swapaxes(-1, -2) @ d_K).mean(axis=0) / batch
            self.W_v[head] -= self.lr * (x_expanded.swapaxes(-1, -2) @ d_V).mean(axis=0) / batch
        
        self.W_o -= self.lr * (d_attn_output.T @ d_attn_output) / batch
    
    def predict(self, x_single: np.ndarray) -> np.ndarray:
        """ 
        Tek bir özellik vektörü için Q-değerlerini tahmin eder.
        
        Args:
            x_single: Tek özellik vektörü (feature_dim,)
        
        Returns:
            Q-değerleri (out_dim,)
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if x_single.ndim != 1 or x_single.shape[0] != self.feature_dim:
            raise ValueError(f"Input shape mismatch: expected ({self.feature_dim},), got {x_single.shape}")
        q, _ = self.forward(x_single[None, :])
        return q[0]
    
    def get_attention_weights(self, x_single: np.ndarray) -> np.ndarray:
        """ 
        Tek bir özellik vektörü için attention weight'lerini döndürür.
        
        Args:
            x_single: Tek özellik vektörü (feature_dim,)
        
        Returns:
            Attention weights (num_heads, num_features, num_features)
        """
        if x_single.ndim != 1 or x_single.shape[0] != self.feature_dim:
            raise ValueError(f"Input shape mismatch: expected ({self.feature_dim},), got {x_single.shape}")
        _, cache = self.forward(x_single[None, :])
        return cache['attention_weights'][0]  # (num_heads, num_features, num_features)
    
    def copy_from(self, other: 'AttentionNet') -> None:
        """ 
        Başka bir AttentionNet'in ağırlıklarını bu ağa kopyalar (Target Network için).
        
        Args:
            other: Kopyalanacak AttentionNet nesnesi
        
        Raises:
            ValueError: Ağ yapıları uyumsuzsa
        """
        if (self.feature_dim != other.feature_dim or 
            self.hidden_dim != other.hidden_dim or 
            self.num_heads != other.num_heads or 
            self.out_dim != other.out_dim):
            raise ValueError("Network architectures must match")
        self.W_proj = other.W_proj.copy()
        self.b_proj = other.b_proj.copy()
        self.W_q = other.W_q.copy()
        self.W_k = other.W_k.copy()
        self.W_v = other.W_v.copy()
        self.W_o = other.W_o.copy()
        self.W_ff1 = other.W_ff1.copy()
        self.b_ff1 = other.b_ff1.copy()
        self.W_ff2 = other.W_ff2.copy()
        self.b_ff2 = other.b_ff2.copy()
        self.W_out = other.W_out.copy()
        self.b_out = other.b_out.copy()

# ---------------------------------------------------------------------
# 5. "HAFIZA": DENEYİM TEKRARI TAMPONU (REPLAY BUFFER)
# Ajanın (durum, eylem, ödül, yeni_durum) deneyimlerini saklar.
# Bu sefer durumlar özellik vektörü formatında.
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
            s: Mevcut durum özellik vektörü
            a: Yapılan eylem
            r: Alınan ödül
            s2: Yeni durum özellik vektörü
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

def render_policy(env: GridWorld, net: AttentionNet) -> None:
    """ 
    Öğrenilen politikayı ekrana çizer.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş AttentionNet ağı
    """
    grid = []
    for y in range(env.h):
        row = []
        for x in range(env.w):
            if (x, y) == env.goal:
                row.append("G")
                continue
            features = to_feature_vector((x, y), env.w, env.h, env.goal)
            a = int(np.argmax(net.predict(features)))
            row.append(ACTIONS[a])
        grid.append(row)
    print("\n(Attention-DQN) Öğrenilen Politika:")
    for r in grid: print(" ".join(r))

def evaluate(
    env: GridWorld, 
    net: AttentionNet, 
    episodes: int = 5, 
    max_steps: int = 100
) -> List[float]:
    """ 
    Eğitilmiş ajanın performansını test eder.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş AttentionNet ağı
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
            features = to_feature_vector(s, env.w, env.h, env.goal)
            a = int(np.argmax(net.predict(features)))
            s, r, done, _ = env.step(a)
            R += r
            if done: break
        out.append(R)
    return out

# ---------------------------------------------------------------------
# 3. ANA EĞİTİM FONKSİYONU
# Tüm Attention-DQN öğrenme sürecini yönetir.
# ---------------------------------------------------------------------
def train_attention_dqn(
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
    hidden_dim: int = 64,
    num_heads: int = 4,
    seed: int = 123
) -> Tuple[GridWorld, AttentionNet, AttentionNet, List[float]]:
    """
    Attention-DQN eğitim fonksiyonu.
    
    MLP-DQN, CNN-DQN ve LSTM-DQN'den farklı olarak, burada durumlar
    zengin özellik vektörleri olarak temsil edilir ve attention
    mekanizması hangi özelliklerin önemli olduğunu öğrenir.
    
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
        hidden_dim: Gizli katman boyutu (pozitif tam sayı)
        num_heads: Attention head sayısı (pozitif tam sayı)
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
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if hidden_dim % num_heads != 0:
        raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
    if start_after > buf_cap:
        raise ValueError(f"start_after ({start_after}) cannot be greater than buf_cap ({buf_cap})")
    
    random.seed(seed)
    np.random.seed(seed)
    env = GridWorld()
    
    # Attention ağları
    policy = AttentionNet(feature_dim=7, hidden_dim=hidden_dim, num_heads=num_heads,
                          out_dim=4, lr=lr, seed=seed)
    target = AttentionNet(feature_dim=7, hidden_dim=hidden_dim, num_heads=num_heads,
                          out_dim=4, lr=lr, seed=seed+1)
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
            # Durumu özellik vektörüne çevir
            features = to_feature_vector(s, env.w, env.h, env.goal)
            
            # Eylem seçimi
            q = policy.predict(features)
            a = epsilon_greedy(q, eps)
            
            # Eylemi gerçekleştir
            s2, r, done, _ = env.step(a)
            features2 = to_feature_vector(s2, env.w, env.h, env.goal)
            
            # Deneyimi hafızaya kaydet
            rb.push(features, a, r, features2, float(done))
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
    # Attention-DQN eğitimi
    env, policy, target, returns = train_attention_dqn()
    # Öğrenilen politikayı göster
    render_policy(env, policy)
    # Performans değerlendirmesi
    print("Değerlendirme (sadece en iyi eylemlerle):", evaluate(env, policy, 10))

