"""
Ortak yardımcı fonksiyonlar.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import random


def to_state_vector(s: Tuple[int, int], w: int, h: int) -> np.ndarray:
    """
    Durumu (x,y) sinir ağına uygun bir vektöre dönüştürür.
    
    Normalizasyon (değerleri 0-1 arasına sıkıştırma), sinir ağının
    daha stabil ve verimli öğrenmesine yardımcı olur.
    
    Args:
        s: Durum koordinatı (x, y)
        w: GridWorld genişliği
        h: GridWorld yüksekliği
    
    Returns:
        Normalize edilmiş durum vektörü [x_norm, y_norm]
        
    Examples:
        >>> to_state_vector((0, 0), 5, 5)
        array([0., 0.], dtype=float32)
        >>> to_state_vector((4, 4), 5, 5)
        array([1., 1.], dtype=float32)
    """
    if w <= 1 or h <= 1:
        raise ValueError(f"GridWorld dimensions must be > 1, got w={w}, h={h}")
    
    x, y = s
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"State {s} is out of bounds for grid size {w}x{h}")
    
    return np.array([x / (w - 1), y / (h - 1)], dtype=np.float32)


def epsilon_greedy(q: np.ndarray, eps: float) -> int:
    """
    ε-greedy eylem seçimi stratejisi.
    
    - `eps` olasılıkla rastgele bir eylem seç (keşfet).
    - `1-eps` olasılıkla en yüksek Q-değerine sahip eylemi seç (kullan).
    
    Args:
        q: Her eylem için Q-değerleri (4 elemanlı dizi: [Q(yukarı), Q(aşağı), Q(sol), Q(sağ)])
        eps: Keşif olasılığı (0.0 ile 1.0 arası)
    
    Returns:
        Seçilen eylem indeksi (0-3 arası)
    
    Raises:
        ValueError: eps değeri 0-1 aralığında değilse veya q dizisi 4 elemanlı değilse
    
    Examples:
        >>> q = np.array([1.0, 2.0, 0.5, 1.5])
        >>> action = epsilon_greedy(q, eps=0.0)  # Sadece kullanım (exploitation)
        >>> assert action == 1  # En yüksek Q-değeri (2.0) indeksi
    """
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"Epsilon must be between 0 and 1, got {eps}")
    
    if q.shape[0] != 4:
        raise ValueError(f"Q-values must have 4 elements (one for each action), got shape {q.shape}")
    
    if random.random() < eps:
        # Keşif: Rastgele eylem seç
        return random.randint(0, 3)
    else:
        # Kullanım: En iyi bilinen eylemi seç
        return int(np.argmax(q))

