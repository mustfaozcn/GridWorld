"""
GridWorld ortamı - Tüm RL algoritmaları için ortak ortam sınıfı.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any


class GridWorld:
    """
    Ajanın içinde hareket ettiği ızgara dünyasını temsil eder.
    
    Bu sınıf, ajanın durumunu (state), yapabileceği eylemleri (actions) ve
    bu eylemlerin sonuçlarını (yeni durum, ödül, bitti mi?) yönetir.
    
    Attributes:
        w: Izgaranın genişliği
        h: Izgaranın yüksekliği
        start: Ajanın başlayacağı koordinat (varsayılan: (0,0))
        goal: Ajanın ulaşmaya çalıştığı hedef koordinat (varsayılan: (4,4))
        state: Ajanın mevcut konumu
        action_space: Ajanın yapabileceği eylemler [0=Yukarı, 1=Aşağı, 2=Sol, 3=Sağ]
    """
    
    def __init__(self, width: int = 5, height: int = 5, start: Tuple[int, int] = (0, 0), goal: Tuple[int, int] = (4, 4)):
        """
        GridWorld ortamını başlatır.
        
        Args:
            width: Izgaranın genişliği (varsayılan: 5)
            height: Izgaranın yüksekliği (varsayılan: 5)
            start: Başlangıç koordinatı (varsayılan: (0,0))
            goal: Hedef koordinatı (varsayılan: (4,4))
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"GridWorld width and height must be positive, got width={width}, height={height}")
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            raise ValueError(f"Start position {start} is out of bounds for grid size {width}x{height}")
        if not (0 <= goal[0] < width and 0 <= goal[1] < height):
            raise ValueError(f"Goal position {goal} is out of bounds for grid size {width}x{height}")
        
        self.w = width
        self.h = height
        self.start = start
        self.goal = goal
        self.state = start
        # Eylemler: 0=Yukarı, 1=Aşağı, 2=Sol, 3=Sağ
        self.action_space = [0, 1, 2, 3]

    def reset(self) -> Tuple[int, int]:
        """
        Ortamı başlangıç durumuna sıfırlar.
        
        Returns:
            Başlangıç durumu (start koordinatı)
        """
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Ajanın bir eylem yapmasını ve sonucunu almasını sağlar.
        
        Args:
            action: Yapılacak eylem (0=Yukarı, 1=Aşağı, 2=Sol, 3=Sağ)
            
        Returns:
            Tuple containing:
                - Yeni durum (x, y)
                - Ödül (reward): Hedefe ulaşırsa +10.0, aksi halde -1.0
                - Bitti mi? (done): Hedefe ulaşıldıysa True
                - Bilgi sözlüğü (info): Ek bilgiler (şu an boş)
        
        Raises:
            ValueError: Geçersiz eylem girildiğinde
        """
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Must be one of {self.action_space}")
        
        x, y = self.state
        
        # Eylemleri uygula
        if action == 0:  # Yukarı
            y = max(0, y - 1)
        elif action == 1:  # Aşağı
            y = min(self.h - 1, y + 1)
        elif action == 2:  # Sol
            x = max(0, x - 1)
        elif action == 3:  # Sağ
            x = min(self.w - 1, x + 1)
        
        self.state = (x, y)

        # Ödül mekanizması
        if self.state == self.goal:
            # Hedefe ulaşırsa +10 ödül ve bölüm biter
            return self.state, 10.0, True, {}
        # Hedefe ulaşamadığı her adım için -1 ceza
        return self.state, -1.0, False, {}

