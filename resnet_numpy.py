
# -*- coding: utf-8 -*-
"""
NumPy ile Sıfırdan ResNet (Residual Network) ile Deep Q-Learning (GridWorld)
-----------------------------------------------------------------------------
- Bu dosya, GridWorld problemini ResNet (Residual Network) kullanarak çözer.
- ResNet'in Gücü: Skip connections ile gradyan akışını iyileştirir,
  daha derin ağları eğitilebilir hale getirir.
- Anahtar Konseptler:
    1. Residual Block: Skip connection içeren bloklar
    2. Skip Connection: Katmanlar arası direkt bağlantı
    3. Gradyan Akışı: Skip connection'lar gradyanları korur
Bu dosyayı tek başına çalıştırabilirsiniz.
"""
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import random

# Ortak modüllerden import
from common import GridWorld, to_state_vector, epsilon_greedy

class ResNet:
    """
    Residual Network (ResNet).
    
    ResNet, skip connections (atlamalı bağlantılar) kullanarak derin ağları
    eğitilebilir hale getirir. Residual block'lar, girdiyi çıktıya direkt
    ekler: output = F(x) + x
    """
    def __init__(
        self, 
        in_dim: int = 2, 
        hidden_dim: int = 64, 
        num_blocks: int = 3, 
        out_dim: int = 4, 
        lr: float = 1e-3, 
        seed: int = 42
    ) -> None:
        """
        ResNet'i başlatır.
        
        Args:
            in_dim: Girdi boyutu
            hidden_dim: Gizli katman boyutu
            num_blocks: Residual block sayısı
            out_dim: Çıktı boyutu
            lr: Öğrenme oranı
            seed: Rastgelelik tohumu
        
        Raises:
            ValueError: Geçersiz parametre değerleri
        """
        if in_dim <= 0 or hidden_dim <= 0 or num_blocks <= 0 or out_dim <= 0:
            raise ValueError(f"All dimensions must be positive, got in_dim={in_dim}, hidden_dim={hidden_dim}, num_blocks={num_blocks}, out_dim={out_dim}")
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.lr = lr
        
        # Giriş projeksiyonu
        self.W_in = rng.standard_normal((in_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0/in_dim)
        self.b_in = np.zeros((hidden_dim,), dtype=np.float32)
        
        # Residual block'lar (her biri 2 katmanlı)
        self.W1_blocks = []
        self.b1_blocks = []
        self.W2_blocks = []
        self.b2_blocks = []
        
        for i in range(num_blocks):
            W1 = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0/hidden_dim)
            b1 = np.zeros((hidden_dim,), dtype=np.float32)
            W2 = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0/hidden_dim)
            b2 = np.zeros((hidden_dim,), dtype=np.float32)
            self.W1_blocks.append(W1)
            self.b1_blocks.append(b1)
            self.W2_blocks.append(W2)
            self.b2_blocks.append(b2)
        
        # Çıkış katmanı
        self.W_out = rng.standard_normal((hidden_dim, out_dim)).astype(np.float32) * np.sqrt(2.0/hidden_dim)
        self.b_out = np.zeros((out_dim,), dtype=np.float32)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)
    
    def residual_block_forward(
        self, 
        x: np.ndarray, 
        block_idx: int
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Residual block: F(x) + x
        Skip connection: Girdiyi direkt çıktıya ekler
        
        Args:
            x: Girdi (batch, hidden_dim)
            block_idx: Block indeksi
        
        Returns:
            Tuple containing:
                - output: Çıktı (batch, hidden_dim)
                - cache: Geri yayılım için ara değerler
        
        Raises:
            IndexError: Block indeksi geçersizse
        """
        if block_idx < 0 or block_idx >= self.num_blocks:
            raise IndexError(f"Block index {block_idx} out of range [0, {self.num_blocks})")
        # İlk katman
        z1 = x @ self.W1_blocks[block_idx] + self.b1_blocks[block_idx]
        a1 = self.relu(z1)
        
        # İkinci katman
        z2 = a1 @ self.W2_blocks[block_idx] + self.b2_blocks[block_idx]
        a2 = self.relu(z2)
        
        # Skip connection: x + a2
        output = x + a2  # RESIDUAL CONNECTION
        
        return output, (x, z1, a1, z2, a2)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        İleri yayılım (Forward Pass).
        
        Args:
            x: Girdi (batch, in_dim)
        
        Returns:
            Tuple containing:
                - q_values: Q-değerleri (batch, out_dim)
                - all_cache: Tüm block'lar için cache'ler
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if x.ndim != 2:
            raise ValueError(f"Input must be 2D (batch, in_dim), got shape {x.shape}")
        if x.shape[1] != self.W_in.shape[0]:
            raise ValueError(f"Input dimension mismatch: expected {self.W_in.shape[0]}, got {x.shape[1]}")
        
        # Giriş projeksiyonu
        h = x @ self.W_in + self.b_in
        h = self.relu(h)
        
        all_cache = []
        
        # Residual block'lar
        for i in range(self.num_blocks):
            h, cache = self.residual_block_forward(h, i)
            all_cache.append(cache)
        
        # Çıkış katmanı
        q = h @ self.W_out + self.b_out
        
        return q, (x, h, all_cache)
    
    def residual_block_backward(self, d_output, cache, block_idx):
        """ Residual block geri yayılımı """
        x, z1, a1, z2, a2 = cache
        
        # Skip connection'dan gelen gradyan: hem d_output'a hem de x'e gider
        d_x = d_output.copy()  # Skip connection gradyanı
        d_a2 = d_output.copy()  # Residual block gradyanı
        
        # İkinci katman geri yayılımı
        d_z2 = d_a2 * self.relu_grad(z2)
        dW2 = a1.T @ d_z2 / d_z2.shape[0]
        db2 = d_z2.mean(axis=0)
        d_a1 = d_z2 @ self.W2_blocks[block_idx].T
        
        # İlk katman geri yayılımı
        d_z1 = d_a1 * self.relu_grad(z1)
        dW1 = x.T @ d_z1 / d_z1.shape[0]
        db1 = d_z1.mean(axis=0)
        d_x_block = d_z1 @ self.W1_blocks[block_idx].T
        
        # Skip connection: x'e giden gradyan
        d_x += d_x_block  # Skip connection + block gradyanı
        
        return d_x, dW1, db1, dW2, db2
    
    def backward(self, cache: Tuple, dq: np.ndarray) -> None:
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
        x, h, all_cache = cache
        batch = x.shape[0]
        
        # Çıkış katmanı
        d_h = dq @ self.W_out.T
        dW_out = h.T @ dq / batch
        db_out = dq.mean(axis=0)
        
        # Residual block'lar (ters sırada)
        for i in reversed(range(self.num_blocks)):
            d_h, dW1, db1, dW2, db2 = self.residual_block_backward(d_h, all_cache[i], i)
            self.W1_blocks[i] -= self.lr * dW1
            self.b1_blocks[i] -= self.lr * db1
            self.W2_blocks[i] -= self.lr * dW2
            self.b2_blocks[i] -= self.lr * db2
        
        # Giriş projeksiyonu
        d_h_in = d_h * self.relu_grad(h)
        dW_in = x.T @ d_h_in / batch
        db_in = d_h_in.mean(axis=0)
        self.W_in -= self.lr * dW_in
        self.b_in -= self.lr * db_in
        self.W_out -= self.lr * dW_out
        self.b_out -= self.lr * db_out
    
    def predict(self, x_single: np.ndarray) -> np.ndarray:
        """ 
        Tek bir durum vektörü için Q-değerlerini tahmin eder.
        
        Args:
            x_single: Tek durum vektörü (in_dim,)
        
        Returns:
            Q-değerleri (out_dim,)
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if x_single.ndim != 1 or x_single.shape[0] != self.W_in.shape[0]:
            raise ValueError(f"Input shape mismatch: expected ({self.W_in.shape[0]},), got {x_single.shape}")
        q, _ = self.forward(x_single[None, :])
        return q[0]
    
    def copy_from(self, other: 'ResNet') -> None:
        """ 
        Başka bir ResNet'in ağırlıklarını bu ağa kopyalar (Target Network için).
        
        Args:
            other: Kopyalanacak ResNet nesnesi
        
        Raises:
            ValueError: Ağ yapıları uyumsuzsa
        """
        if (self.W_in.shape[0] != other.W_in.shape[0] or 
            self.hidden_dim != other.hidden_dim or 
            self.num_blocks != other.num_blocks or 
            self.out_dim != other.out_dim):
            raise ValueError("Network architectures must match")
        
        self.W_in = other.W_in.copy()
        self.b_in = other.b_in.copy()
        for i in range(self.num_blocks):
            self.W1_blocks[i] = other.W1_blocks[i].copy()
            self.b1_blocks[i] = other.b1_blocks[i].copy()
            self.W2_blocks[i] = other.W2_blocks[i].copy()
            self.b2_blocks[i] = other.b2_blocks[i].copy()
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
    
    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: float) -> None:
        """ 
        Bir deneyimi hafızaya ekler.
        
        Args:
            s: Mevcut durum vektörü
            a: Yapılan eylem
            r: Alınan ödül
            s2: Yeni durum vektörü
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

ACTIONS = {0:"↑", 1:"↓", 2:"←", 3:"→"}

def render_policy(env: GridWorld, net: ResNet) -> None:
    """ 
    Öğrenilen politikayı ekrana çizer.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş ResNet ağı
    """
    grid: List[List[str]] = []
    for y in range(env.h):
        row = []
        for x in range(env.w):
            if (x, y) == env.goal:
                row.append("G")
                continue
            s = to_state_vector((x, y), env.w, env.h)
            a = int(np.argmax(net.predict(s)))
            row.append(ACTIONS[a])
        grid.append(row)
    print("\n(ResNet-DQN) Öğrenilen Politika:")
    for r in grid: print(" ".join(r))

def evaluate(
    env: GridWorld, 
    net: ResNet, 
    episodes: int = 5, 
    max_steps: int = 100
) -> List[float]:
    """ 
    Eğitilmiş ajanın performansını test eder.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş ResNet ağı
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
            a = int(np.argmax(net.predict(to_state_vector(s, env.w, env.h))))
            s, r, done, _ = env.step(a); R += r
            if done: break
        out.append(R)
    return out

def train_resnet_dqn(
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
    num_blocks: int = 3,
    hidden_dim: int = 64,
    seed: int = 123
) -> Tuple[GridWorld, ResNet, ResNet, List[float]]:
    """
    ResNet-DQN eğitim fonksiyonu.
    
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
        num_blocks: Residual block sayısı (pozitif tam sayı)
        hidden_dim: Gizli katman boyutu (pozitif tam sayı)
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
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if start_after > buf_cap:
        raise ValueError(f"start_after ({start_after}) cannot be greater than buf_cap ({buf_cap})")
    
    random.seed(seed)
    np.random.seed(seed)
    env = GridWorld()
    policy = ResNet(2, hidden_dim, num_blocks, 4, lr, seed)
    target = ResNet(2, hidden_dim, num_blocks, 4, lr, seed+1)
    target.copy_from(policy)
    rb = ReplayBuffer(buf_cap)
    eps = eps_start; returns = []; steps = 0
    
    for ep in range(episodes):
        s = env.reset(); epR = 0.0
        for t in range(max_steps):
            s_vec = to_state_vector(s, env.w, env.h)
            q = policy.predict(s_vec)
            a = epsilon_greedy(q, eps)
            s2, r, done, _ = env.step(a)
            rb.push(s_vec, a, r, to_state_vector(s2, env.w, env.h), float(done))
            epR += r; s = s2; steps += 1
            if len(rb) >= max(batch, start_after):
                S, A, R, S2, D = rb.sample(batch)
                Qs, cache = policy.forward(S)
                if use_target: Qs2, _ = target.forward(S2)
                else: Qs2, _ = policy.forward(S2)
                y = R + gamma * (1.0 - D) * np.max(Qs2, axis=1)
                dq = np.zeros_like(Qs)
                idx = np.arange(batch)
                dq[idx, A] = (Qs[idx, A] - y)
                policy.backward(cache, dq)
            if done: break
            if use_target and steps % target_every == 0: target.copy_from(policy)
        returns.append(epR)
        eps = max(eps_min, eps * eps_decay)
    return env, policy, target, returns

if __name__ == "__main__":
    env, policy, target, returns = train_resnet_dqn()
    render_policy(env, policy)
    print("Değerlendirme:", evaluate(env, policy, 10))

