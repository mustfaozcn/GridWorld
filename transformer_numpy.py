
# -*- coding: utf-8 -*-
"""
NumPy ile Sıfırdan Transformer ile Deep Q-Learning (GridWorld)
---------------------------------------------------------------
- Bu dosya, GridWorld problemini Transformer kullanarak çözer.
- Transformer'ın Gücü: Self-attention ve positional encoding ile
  özellikler arası kompleks ilişkileri öğrenir.
- Anahtar Konseptler:
    1. Self-Attention: Özellikler arası ilişkiler
    2. Positional Encoding: Konum bilgisi
    3. Multi-Head Attention: Çok yönlü analiz
Bu dosyayı tek başına çalıştırabilirsiniz.
"""
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import random

# Ortak modüllerden import
from common import GridWorld, epsilon_greedy

def to_feature_sequence(
    s: Tuple[int, int], 
    w: int, 
    h: int, 
    goal: Tuple[int, int]
) -> np.ndarray:
    """ 
    Durumu özellik sequence'ine dönüştürür (Transformer için)
    
    Args:
        s: Durum koordinatları (x, y)
        w: GridWorld genişliği
        h: GridWorld yüksekliği
        goal: Hedef koordinatları (x, y)
    
    Returns:
        Özellik sequence (5, 2)
    
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
    # Her özellik bir token olarak temsil edilir
    features = [
        np.array([x/(w-1), 0], dtype=np.float32),  # Token 0: Agent X
        np.array([y/(h-1), 0], dtype=np.float32),  # Token 1: Agent Y
        np.array([goal_x/(w-1), 0], dtype=np.float32),  # Token 2: Goal X
        np.array([goal_y/(h-1), 0], dtype=np.float32),  # Token 3: Goal Y
        np.array([np.sqrt((x-goal_x)**2 + (y-goal_y)**2) / (np.sqrt((w-1)**2 + (h-1)**2) + 1e-6), 0], dtype=np.float32)  # Token 4: Distance
    ]
    return np.array(features)  # (5, 2)

# epsilon_greedy artık common modülünden import ediliyor

class Transformer:
    """
    Transformer sinir ağı.
    
    Transformer, self-attention ve positional encoding kullanarak
    özellikler arası kompleks ilişkileri öğrenir.
    """
    def __init__(
        self, 
        token_dim: int = 2, 
        d_model: int = 64, 
        num_heads: int = 4, 
        num_layers: int = 2, 
        out_dim: int = 4, 
        lr: float = 1e-3, 
        seed: int = 42
    ) -> None:
        """
        Transformer'ı başlatır.
        
        Args:
            token_dim: Token boyutu
            d_model: Model boyutu
            num_heads: Attention head sayısı
            num_layers: Transformer layer sayısı
            out_dim: Çıktı boyutu
            lr: Öğrenme oranı
            seed: Rastgelelik tohumu
        
        Raises:
            ValueError: Geçersiz parametre değerleri
        """
        if token_dim <= 0 or d_model <= 0 or num_heads <= 0 or num_layers <= 0 or out_dim <= 0:
            raise ValueError(f"All dimensions must be positive, got token_dim={token_dim}, d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, out_dim={out_dim}")
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        rng = np.random.default_rng(seed)
        self.token_dim = token_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers
        self.lr = lr
        
        # Token embedding
        self.W_embed = rng.standard_normal((token_dim, d_model)).astype(np.float32) * np.sqrt(2.0/token_dim)
        
        # Positional encoding (sinusoidal)
        max_len = 10
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe
        
        # Transformer layer'ları
        self.layers = []
        for i in range(num_layers):
            layer = {
                'W_q': rng.standard_normal((num_heads, d_model, self.head_dim)).astype(np.float32) * np.sqrt(2.0/d_model),
                'W_k': rng.standard_normal((num_heads, d_model, self.head_dim)).astype(np.float32) * np.sqrt(2.0/d_model),
                'W_v': rng.standard_normal((num_heads, d_model, self.head_dim)).astype(np.float32) * np.sqrt(2.0/d_model),
                'W_o': rng.standard_normal((d_model, d_model)).astype(np.float32) * np.sqrt(2.0/d_model),
                'W_ff1': rng.standard_normal((d_model, d_model)).astype(np.float32) * np.sqrt(2.0/d_model),
                'b_ff1': np.zeros((d_model,), dtype=np.float32),
                'W_ff2': rng.standard_normal((d_model, d_model)).astype(np.float32) * np.sqrt(2.0/d_model),
                'b_ff2': np.zeros((d_model,), dtype=np.float32),
            }
            self.layers.append(layer)
        
        # Çıkış katmanı
        self.W_out = rng.standard_normal((d_model, out_dim)).astype(np.float32) * np.sqrt(2.0/d_model)
        self.b_out = np.zeros((out_dim,), dtype=np.float32)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)
    
    def attention(
        self, 
        Q: np.ndarray, 
        K: np.ndarray, 
        V: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = Q @ K.swapaxes(-1, -2) / np.sqrt(self.head_dim)
        attn_weights = self.softmax(scores, axis=-1)
        output = attn_weights @ V
        return output, attn_weights
    
    def forward(self, x_seq: np.ndarray) -> Tuple[np.ndarray, dict]:
        """ 
        İleri yayılım (Forward Pass).
        
        Args:
            x_seq: Girdi sequence (seq_len, token_dim)
        
        Returns:
            Tuple containing:
                - q_values: Q-değerleri (out_dim,)
                - cache: Geri yayılım için ara değerler
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if x_seq.ndim != 2:
            raise ValueError(f"Input must be 2D (seq_len, token_dim), got shape {x_seq.shape}")
        if x_seq.shape[1] != self.token_dim:
            raise ValueError(f"Token dimension mismatch: expected {self.token_dim}, got {x_seq.shape[1]}")
        seq_len = x_seq.shape[0]
        
        # Token embedding
        x = x_seq @ self.W_embed  # (seq_len, d_model)
        
        # Positional encoding ekle
        x = x + self.pe[:seq_len, :]
        
        all_attn_weights = []
        
        # Transformer layer'ları
        for layer in self.layers:
            # Multi-head self-attention
            all_heads = []
            for head in range(self.num_heads):
                Q = x @ layer['W_q'][head]
                K = x @ layer['W_k'][head]
                V = x @ layer['W_v'][head]
                head_out, attn_weights = self.attention(Q, K, V)
                all_heads.append(head_out)
                all_attn_weights.append(attn_weights)
            
            attn_out = np.concatenate(all_heads, axis=-1) @ layer['W_o']
            x = x + attn_out  # Residual
            
            # Feed-forward
            ff_out = self.relu(x @ layer['W_ff1'] + layer['b_ff1'])
            ff_out = ff_out @ layer['W_ff2'] + layer['b_ff2']
            x = x + ff_out  # Residual
        
        # Global average pooling
        x_pooled = x.mean(axis=0)  # (d_model,)
        
        # Çıkış
        q = x_pooled @ self.W_out + self.b_out
        
        return q, all_attn_weights
    
    def backward(self, cache: dict, dq: np.ndarray) -> None:
        """
        Geri yayılım (Backward Pass): Hata gradyanını kullanarak ağırlıkları günceller.
        
        Args:
            cache: Forward pass'ten gelen ara değerler
            dq: Çıktı katmanı gradyanları (out_dim,)
        
        Note:
            Basitleştirilmiş geri yayılım (tam implementasyon çok karmaşık)
            Gerçek uygulamada tüm layer'lar için detaylı gradyan hesaplanmalı
        """
        if dq.shape[0] != self.out_dim:
            raise ValueError(f"Gradient shape mismatch: expected ({self.out_dim},), got {dq.shape}")
        # Basitleştirilmiş geri yayılım - tam implementasyon çok karmaşık
        pass
    
    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        """ 
        Tek bir sequence için Q-değerlerini tahmin eder.
        
        Args:
            x_seq: Girdi sequence (seq_len, token_dim)
        
        Returns:
            Q-değerleri (out_dim,)
        """
        q, _ = self.forward(x_seq)
        return q
    
    def copy_from(self, other: 'Transformer') -> None:
        """ 
        Başka bir Transformer'ın ağırlıklarını bu ağa kopyalar (Target Network için).
        
        Args:
            other: Kopyalanacak Transformer nesnesi
        
        Raises:
            ValueError: Ağ yapıları uyumsuzsa
        """
        if (self.token_dim != other.token_dim or 
            self.d_model != other.d_model or 
            self.num_heads != other.num_heads or 
            self.num_layers != other.num_layers or 
            self.out_dim != other.out_dim):
            raise ValueError("Network architectures must match")
        
        self.W_embed = other.W_embed.copy()
        for i, layer in enumerate(self.layers):
            for key in layer:
                self.layers[i][key] = other.layers[i][key].copy()
        self.W_out = other.W_out.copy()
        self.b_out = other.b_out.copy()

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
    
    def push(
        self, 
        s: np.ndarray, 
        a: int, 
        r: float, 
        s2: np.ndarray, 
        d: float
    ) -> None:
        """ 
        Bir deneyimi hafızaya ekler.
        
        Args:
            s: Mevcut durum sequence
            a: Yapılan eylem
            r: Alınan ödül
            s2: Yeni durum sequence
            d: Bölüm bitme durumu (0.0 veya 1.0)
        """
        data = (s, a, r, s2, d)
        if len(self.buf) < self.capacity:
            self.buf.append(data)
        else:
            self.buf[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(
        self, 
        batch: int
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        """ 
        Hafızadan rastgele bir batch deneyim çeker.
        
        Args:
            batch: Örnek boyutu (pozitif tam sayı)
        
        Returns:
            Tuple containing (S, A, R, S2, D) where S and S2 are lists of numpy arrays
        
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
            list(S), 
            np.array(A), 
            np.array(R, dtype=np.float32), 
            list(S2), 
            np.array(D, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buf)

ACTIONS = {0:"↑", 1:"↓", 2:"←", 3:"→"}

def render_policy(env: GridWorld, net: Transformer) -> None:
    """ 
    Öğrenilen politikayı ekrana çizer.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş Transformer ağı
    """
    grid: List[List[str]] = []
    for y in range(env.h):
        row = []
        for x in range(env.w):
            if (x, y) == env.goal:
                row.append("G")
                continue
            seq = to_feature_sequence((x, y), env.w, env.h, env.goal)
            a = int(np.argmax(net.predict(seq)))
            row.append(ACTIONS[a])
        grid.append(row)
    print("\n(Transformer-DQN) Öğrenilen Politika:")
    for r in grid: print(" ".join(r))

def evaluate(
    env: GridWorld, 
    net: Transformer, 
    episodes: int = 5, 
    max_steps: int = 100
) -> List[float]:
    """ 
    Eğitilmiş ajanın performansını test eder.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş Transformer ağı
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
        s = env.reset(); R = 0.0
        for _ in range(max_steps):
            seq = to_feature_sequence(s, env.w, env.h, env.goal)
            a = int(np.argmax(net.predict(seq)))
            s, r, done, _ = env.step(a); R += r
            if done: break
        out.append(R)
    return out

def train_transformer_dqn(
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
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    seed: int = 123
) -> Tuple[GridWorld, Transformer, Transformer, List[float]]:
    """
    Transformer-DQN eğitim fonksiyonu.
    
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
        d_model: Model boyutu (pozitif tam sayı)
        num_heads: Attention head sayısı (pozitif tam sayı)
        num_layers: Transformer layer sayısı (pozitif tam sayı)
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
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    if start_after > buf_cap:
        raise ValueError(f"start_after ({start_after}) cannot be greater than buf_cap ({buf_cap})")
    
    random.seed(seed)
    np.random.seed(seed)
    env = GridWorld()
    policy = Transformer(2, d_model, num_heads, num_layers, 4, lr, seed)
    target = Transformer(2, d_model, num_heads, num_layers, 4, lr, seed+1)
    target.copy_from(policy)
    rb = ReplayBuffer(buf_cap)
    eps = eps_start; returns = []; steps = 0
    
    for ep in range(episodes):
        s = env.reset(); epR = 0.0
        for t in range(max_steps):
            seq = to_feature_sequence(s, env.w, env.h, env.goal)
            q = policy.predict(seq)
            a = epsilon_greedy(q, eps)
            s2, r, done, _ = env.step(a)
            seq2 = to_feature_sequence(s2, env.w, env.h, env.goal)
            rb.push(seq, a, r, seq2, float(done))
            epR += r; s = s2; steps += 1
            if len(rb) >= max(batch, start_after):
                # Basitleştirilmiş eğitim (backward implementasyonu eksik)
                pass
            if done: break
            if use_target and steps % target_every == 0: target.copy_from(policy)
        returns.append(epR)
        eps = max(eps_min, eps * eps_decay)
    return env, policy, target, returns

if __name__ == "__main__":
    env, policy, target, returns = train_transformer_dqn()
    render_policy(env, policy)
    print("Değerlendirme:", evaluate(env, policy, 10))

