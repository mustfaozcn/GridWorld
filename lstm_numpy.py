
# -*- coding: utf-8 -*-
"""
NumPy ile Sıfırdan LSTM (Long Short-Term Memory) ile Deep Q-Learning (GridWorld)
----------------------------------------------------------------------------------
- Bu dosya, GridWorld problemini LSTM (Long Short-Term Memory) kullanarak çözer.
- MLP ve CNN'den farklı olarak, LSTM geçmiş adımları hatırlar ve bu bilgiyi
  kullanarak daha iyi kararlar verir.
- LSTM'in Gücü: Geçmiş hareketleri hafızada tutarak döngülerden kaçınır,
  daha uzun vadeli stratejiler geliştirir.
- Anahtar Konseptler:
    1. Sequence Input: Son N adımın durumlarını bir sequence olarak alır.
    2. LSTM Hücresi: Forget gate, Input gate, Output gate ile hafıza yönetimi.
    3. Cell State: Uzun vadeli bilgileri saklar.
    4. Hidden State: Kısa vadeli bilgileri taşır.
Bu dosyayı tek başına çalıştırabilirsiniz.
"""
# ---------------------------------------------------------------------
# OKUMA SIRASI TAVSİYESİ
# ---------------------------------------------------------------------
# 1. `GridWorld` Sınıfı: Ajanın içinde yaşadığı ortam.
# 2. `to_state_vector` Fonksiyonu: Durumu vektöre dönüştürme.
# 3. `if __name__ == "__main__"` Bloğu: Kodun ana akışı.
# 4. `train_lstm_dqn` Fonksiyonu: LSTM-DQN eğitim sürecini yöneten ana döngü.
# 5. `LSTM` Sınıfı: LSTM sinir ağı (beyin).
# 6. `ReplayBuffer` Sınıfı: Ajanın deneyimlerini sakladığı hafıza.
# 7. Diğer yardımcı fonksiyonlar (`epsilon_greedy` vb.).
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, List
import numpy as np
import random

# Ortak modüllerden import
from common import GridWorld, to_state_vector, epsilon_greedy

# to_state_vector ve epsilon_greedy artık common modülünden import ediliyor

# ---------------------------------------------------------------------
# 4. "BEYİN": LSTM (LONG SHORT-TERM MEMORY)
# Q-değerlerini tahmin etmek için kullanılan LSTM sinir ağı.
# LSTM'in avantajı, geçmiş adımları hatırlayabilmesi ve bu bilgiyi
# kullanarak daha iyi kararlar verebilmesidir. Özellikle döngülerden
# kaçınma ve uzun vadeli strateji geliştirme konusunda üstündür.
# ---------------------------------------------------------------------
class LSTM:
    """
    Long Short-Term Memory (LSTM) sinir ağı.
    
    LSTM, geçmiş bilgileri hatırlayabilen özel bir sinir ağı türüdür.
    Bu, GridWorld'de "5 adım önce neredeydim?" sorusunu cevaplayabilmesi
    anlamına gelir. Bu sayede ajan, aynı yere tekrar tekrar gitmekten
    (döngülerden) kaçınabilir.
    """
    def __init__(
        self, 
        input_dim: int = 2, 
        hidden_dim: int = 64, 
        sequence_length: int = 10, 
        out_dim: int = 4, 
        lr: float = 1e-3, 
        seed: int = 42
    ) -> None:
        """
        LSTM'in başlatılması.
        
        Args:
            input_dim: Her adımda gelen durum vektörünün boyutu (2: x, y)
            hidden_dim: LSTM hücresinin gizli durum boyutu
            sequence_length: Kaç adım geriye bakılacağı (örn: 10)
            out_dim: Çıktı boyutu (Q-değerleri için 4 eylem)
            lr: Öğrenme oranı
            seed: Rastgele sayı üreteci için tohum
        
        Raises:
            ValueError: Geçersiz parametre değerleri
        """
        if input_dim <= 0 or hidden_dim <= 0 or sequence_length <= 0 or out_dim <= 0:
            raise ValueError(f"All dimensions must be positive, got input_dim={input_dim}, hidden_dim={hidden_dim}, sequence_length={sequence_length}, out_dim={out_dim}")
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = sequence_length
        self.lr = lr
        
        # LSTM'in 4 gate'i için ağırlıklar:
        # 1. Forget Gate (f_t): "Eski bilgiyi unutmalı mıyım?"
        # 2. Input Gate (i_t): "Yeni bilgiyi hafızaya kaydetmeli miyim?"
        # 3. Candidate Gate (g_t): "Yeni bilginin değeri nedir?"
        # 4. Output Gate (o_t): "Hangi bilgiyi dışarı çıkarayım?"
        
        # Her gate, mevcut girdi (x_t) ve önceki hidden state (h_{t-1}) alır.
        # Toplam girdi boyutu: input_dim + hidden_dim
        gate_input_size = input_dim + hidden_dim
        
        # Forget Gate ağırlıkları
        self.W_f = rng.standard_normal((gate_input_size, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / gate_input_size)
        self.b_f = np.zeros((hidden_dim,), dtype=np.float32)
        
        # Input Gate ağırlıkları
        self.W_i = rng.standard_normal((gate_input_size, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / gate_input_size)
        self.b_i = np.zeros((hidden_dim,), dtype=np.float32)
        
        # Candidate Gate (cell state için yeni bilgi) ağırlıkları
        self.W_g = rng.standard_normal((gate_input_size, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / gate_input_size)
        self.b_g = np.zeros((hidden_dim,), dtype=np.float32)
        
        # Output Gate ağırlıkları
        self.W_o = rng.standard_normal((gate_input_size, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / gate_input_size)
        self.b_o = np.zeros((hidden_dim,), dtype=np.float32)
        
        # Çıkış katmanı: Hidden state'i Q-değerlerine dönüştürür
        self.W_out = rng.standard_normal((hidden_dim, out_dim)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros((out_dim,), dtype=np.float32)
    
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid aktivasyon fonksiyonu.
        
        Sigmoid, değerleri 0 ile 1 arasına sıkıştırır. Bu, gate'lerin
        "ne kadar açık/kapalı" olduğunu temsil eder:
        - 0: Kapalı (bilgi geçmez)
        - 1: Tam açık (tüm bilgi geçer)
        - 0.5: Yarım açık (bilginin yarısı geçer)
        
        Matematiksel olarak: sigmoid(x) = 1 / (1 + exp(-x))
        """
        # Sayısal stabilite için: exp(-x) yerine clamp kullan
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))
    
    @staticmethod
    def sigmoid_grad(x):
        """ Sigmoid'in türevi (geri yayılım için). """
        s = LSTM.sigmoid(x)
        return s * (1.0 - s)
    
    @staticmethod
    def tanh(x):
        """
        Tanh aktivasyon fonksiyonu.
        
        Tanh, değerleri -1 ile 1 arasına sıkıştırır. Bu, cell state'in
        değerlerini normalleştirmek için kullanılır. Tanh, sigmoid'den
        farklı olarak negatif değerlere de izin verir, bu sayede hem
        pozitif hem negatif bilgileri saklayabiliriz.
        """
        x_clipped = np.clip(x, -500, 500)
        return np.tanh(x_clipped)
    
    @staticmethod
    def tanh_grad(x):
        """ Tanh'in türevi (geri yayılım için). """
        t = LSTM.tanh(x)
        return 1.0 - t * t
    
    def forward_step(self, x_t, h_prev, c_prev):
        """
        LSTM'in bir zaman adımı için ileri yayılımı.
        
        Parametreler:
        - x_t: Mevcut zaman adımındaki girdi (durum vektörü)
        - h_prev: Önceki zaman adımındaki hidden state
        - c_prev: Önceki zaman adımındaki cell state
        
        Çıktı:
        - h_t: Mevcut zaman adımındaki hidden state
        - c_t: Mevcut zaman adımındaki cell state
        - cache: Geri yayılım için saklanacak ara değerler
        
        LSTM'in çalışma mantığı:
        1. Forget Gate: Eski cell state'ten ne kadarını unutmalıyım?
        2. Input Gate: Yeni bilgiyi ne kadar hafızaya almalıyım?
        3. Candidate: Yeni bilginin değeri nedir?
        4. Cell State: Eski bilgiyi unut, yeni bilgiyi ekle
        5. Output Gate: Cell state'ten ne kadarını dışarı çıkarayım?
        6. Hidden State: Output gate ile filtrelenmiş cell state
        """
        # Girdiyi birleştir: [mevcut_durum, önceki_hidden_state]
        concat_input = np.concatenate([x_t, h_prev], axis=-1)  # (batch, input_dim + hidden_dim)
        
        # --- FORGET GATE (f_t) ---
        # "Eski bilgiyi ne kadar unutmalıyım?"
        # Eğer forget gate 0'a yakınsa: "Eski bilgiyi unut"
        # Eğer forget gate 1'e yakınsa: "Eski bilgiyi koru"
        f_t = self.sigmoid(concat_input @ self.W_f + self.b_f)  # (batch, hidden_dim)
        
        # --- INPUT GATE (i_t) ---
        # "Yeni bilgiyi ne kadar hafızaya almalıyım?"
        # Eğer input gate 0'a yakınsa: "Yeni bilgiyi görmezden gel"
        # Eğer input gate 1'e yakınsa: "Yeni bilgiyi hafızaya kaydet"
        i_t = self.sigmoid(concat_input @ self.W_i + self.b_i)  # (batch, hidden_dim)
        
        # --- CANDIDATE GATE (g_t) ---
        # "Yeni bilginin değeri nedir?"
        # Bu, input gate ile çarpılarak cell state'e eklenir.
        # Tanh kullanılır çünkü hem pozitif hem negatif değerlere izin verir.
        g_t = self.tanh(concat_input @ self.W_g + self.b_g)  # (batch, hidden_dim)
        
        # --- CELL STATE GÜNCELLEMESİ (c_t) ---
        # Cell state, LSTM'in "uzun vadeli hafızasıdır".
        # Formül: c_t = f_t * c_prev + i_t * g_t
        # - f_t * c_prev: Eski bilgiyi forget gate ile filtrelenmiş şekilde koru
        # - i_t * g_t: Yeni bilgiyi input gate ile filtrelenmiş şekilde ekle
        c_t = f_t * c_prev + i_t * g_t  # (batch, hidden_dim)
        
        # --- OUTPUT GATE (o_t) ---
        # "Cell state'ten ne kadarını dışarı çıkarayım?"
        # Output gate, cell state'in hangi kısmının hidden state olacağını belirler.
        o_t = self.sigmoid(concat_input @ self.W_o + self.b_o)  # (batch, hidden_dim)
        
        # --- HIDDEN STATE (h_t) ---
        # Hidden state, LSTM'in "kısa vadeli hafızasıdır".
        # Formül: h_t = o_t * tanh(c_t)
        # - tanh(c_t): Cell state'i normalize et (-1 ile 1 arasına sıkıştır)
        # - o_t * tanh(c_t): Output gate ile filtrelenmiş cell state
        h_t = o_t * self.tanh(c_t)  # (batch, hidden_dim)
        
        # Cache: Geri yayılım için ara değerleri sakla
        cache = {
            'x_t': x_t,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'concat_input': concat_input,
            'f_t': f_t,
            'i_t': i_t,
            'g_t': g_t,
            'c_t': c_t,
            'o_t': o_t,
            'h_t': h_t
        }
        
        return h_t, c_t, cache
    
    def forward(self, X_sequence: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        LSTM'in tüm sequence için ileri yayılımı.
        
        Args:
            X_sequence: (batch, sequence_length, input_dim) boyutunda sequence
        
        Returns:
            Tuple containing:
                - q_values: (batch, out_dim) Q-değerleri
                - all_cache: Tüm zaman adımları için cache'ler
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if X_sequence.ndim != 3:
            raise ValueError(f"Input must be 3D (batch, sequence_length, input_dim), got shape {X_sequence.shape}")
        batch, seq_len, input_dim = X_sequence.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {input_dim}")
        if seq_len != self.seq_len:
            raise ValueError(f"Sequence length mismatch: expected {self.seq_len}, got {seq_len}")
        
        # Hidden state ve cell state'i sıfırla (başlangıç durumu)
        h_t = np.zeros((batch, self.hidden_dim), dtype=np.float32)
        c_t = np.zeros((batch, self.hidden_dim), dtype=np.float32)
        
        # Tüm zaman adımları için cache'leri sakla
        all_cache = []
        
        # Sequence boyunca ileri yayılım
        for t in range(seq_len):
            x_t = X_sequence[:, t, :]  # (batch, input_dim)
            h_t, c_t, cache = self.forward_step(x_t, h_t, c_t)
            all_cache.append(cache)
        
        # Son hidden state'i Q-değerlerine dönüştür
        q_values = h_t @ self.W_out + self.b_out  # (batch, out_dim)
        
        return q_values, all_cache
    
    def backward_step(self, dh_next, dc_next, cache):
        """
        LSTM'in bir zaman adımı için geri yayılımı.
        
        LSTM'in geri yayılımı karmaşıktır çünkü:
        1. Hem önceki hidden state'e (h_prev) hem de sonraki hidden state'e (h_next) bağlıdır
        2. Cell state'ten geçen gradyanlar unutma mekanizmasına göre filtrelenir
        3. Her gate'in kendi gradyanı vardır
        
        Parametreler:
        - dh_next: Sonraki zaman adımından gelen hidden state gradyanı
        - dc_next: Sonraki zaman adımından gelen cell state gradyanı
        - cache: Bu zaman adımı için saklanmış ara değerler
        
        Çıktı:
        - dh_prev: Önceki hidden state'e gönderilecek gradyan
        - dc_prev: Önceki cell state'e gönderilecek gradyan
        - dW_f, dW_i, dW_g, dW_o: Gate ağırlıklarının gradyanları
        - db_f, db_i, db_g, db_o: Gate bias'larının gradyanları
        """
        # Cache'den değerleri al
        x_t = cache['x_t']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        concat_input = cache['concat_input']
        f_t = cache['f_t']
        i_t = cache['i_t']
        g_t = cache['g_t']
        c_t = cache['c_t']
        o_t = cache['o_t']
        h_t = cache['h_t']
        
        # Hidden state gradyanı: hem output gate'den hem de sonraki adımdan gelir
        # (eğer son adımsa, dh_next = 0 olabilir)
        dh_t = dh_next.copy() if dh_next is not None else np.zeros_like(h_t)
        
        # Cell state gradyanı: hem forget gate'den hem de output gate'den gelir
        dc_t = dc_next.copy() if dc_next is not None else np.zeros_like(c_t)
        
        # Output gate'den gelen gradyan
        # h_t = o_t * tanh(c_t) olduğu için:
        # dh_t / do_t = tanh(c_t)
        # dh_t / dc_t = o_t * (1 - tanh(c_t)^2) = o_t * tanh_grad(c_t)
        do_t = dh_t * self.tanh(c_t)  # (batch, hidden_dim)
        dc_t += dh_t * o_t * self.tanh_grad(c_t)  # (batch, hidden_dim)
        
        # Output gate gradyanları
        do_t_act = do_t * self.sigmoid_grad(concat_input @ self.W_o + self.b_o)
        dW_o = concat_input.T @ do_t_act / do_t.shape[0]
        db_o = do_t_act.mean(axis=0)
        dconcat_o = do_t_act @ self.W_o.T
        
        # Cell state'ten gelen gradyan
        # c_t = f_t * c_prev + i_t * g_t olduğu için:
        # dc_t / df_t = c_prev
        # dc_t / di_t = g_t
        # dc_t / dg_t = i_t
        df_t = dc_t * c_prev  # (batch, hidden_dim)
        dg_t = dc_t * i_t  # (batch, hidden_dim)
        di_t = dc_t * g_t  # (batch, hidden_dim)
        dc_prev = dc_t * f_t  # (batch, hidden_dim)
        
        # Forget gate gradyanları
        df_t_act = df_t * self.sigmoid_grad(concat_input @ self.W_f + self.b_f)
        dW_f = concat_input.T @ df_t_act / df_t.shape[0]
        db_f = df_t_act.mean(axis=0)
        dconcat_f = df_t_act @ self.W_f.T
        
        # Input gate gradyanları
        di_t_act = di_t * self.sigmoid_grad(concat_input @ self.W_i + self.b_i)
        dW_i = concat_input.T @ di_t_act / di_t.shape[0]
        db_i = di_t_act.mean(axis=0)
        dconcat_i = di_t_act @ self.W_i.T
        
        # Candidate gate gradyanları
        dg_t_act = dg_t * self.tanh_grad(concat_input @ self.W_g + self.b_g)
        dW_g = concat_input.T @ dg_t_act / dg_t.shape[0]
        db_g = dg_t_act.mean(axis=0)
        dconcat_g = dg_t_act @ self.W_g.T
        
        # Concat input gradyanları (tüm gate'lerden gelen gradyanları topla)
        dconcat = dconcat_f + dconcat_i + dconcat_g + dconcat_o
        
        # Girdi ve hidden state gradyanlarına ayır
        input_dim = x_t.shape[-1]
        dx_t = dconcat[:, :input_dim]
        dh_prev = dconcat[:, input_dim:] + dh_t  # Hem concat'ten hem de output gate'den
        
        return dx_t, dh_prev, dc_prev, dW_f, db_f, dW_i, db_i, dW_g, db_g, dW_o, db_o
    
    def backward(self, X_sequence: np.ndarray, all_cache: List, dq: np.ndarray) -> None:
        """
        LSTM'in tüm sequence için geri yayılımı.
        
        Geri yayılım, zaman içinde ters sırada yapılır (Backpropagation Through Time - BPTT).
        Son adımdan başlayıp ilk adıma doğru gideriz.
        
        Args:
            X_sequence: Girdi sequence (batch, sequence_length, input_dim)
            all_cache: Forward pass'ten gelen tüm cache'ler
            dq: Çıktı katmanı gradyanları (batch, out_dim)
        
        Raises:
            ValueError: Gradyan boyutu beklenen boyutla eşleşmiyorsa
        """
        if dq.shape[1] != self.out_dim:
            raise ValueError(f"Gradient shape mismatch: expected (batch, {self.out_dim}), got {dq.shape}")
        
        batch, seq_len, input_dim = X_sequence.shape
        
        # Çıkış katmanı gradyanları
        # Son hidden state'i al
        h_last = all_cache[-1]['h_t']
        dh_last = dq @ self.W_out.T  # (batch, hidden_dim)
        dW_out = h_last.T @ dq / batch  # (hidden_dim, out_dim)
        db_out = dq.mean(axis=0)
        
        # Zaman içinde geri yayılım
        dc_next = None
        dh_next = dh_last
        
        # Gate gradyanlarını biriktir
        dW_f_sum = np.zeros_like(self.W_f)
        db_f_sum = np.zeros_like(self.b_f)
        dW_i_sum = np.zeros_like(self.W_i)
        db_i_sum = np.zeros_like(self.b_i)
        dW_g_sum = np.zeros_like(self.W_g)
        db_g_sum = np.zeros_like(self.b_g)
        dW_o_sum = np.zeros_like(self.W_o)
        db_o_sum = np.zeros_like(self.b_o)
        
        # Son adımdan ilk adıma doğru geri yayılım
        for t in reversed(range(seq_len)):
            dx_t, dh_next, dc_next, dW_f, db_f, dW_i, db_i, dW_g, db_g, dW_o, db_o = \
                self.backward_step(dh_next, dc_next, all_cache[t])
            
            # Gradyanları topla
            dW_f_sum += dW_f
            db_f_sum += db_f
            dW_i_sum += dW_i
            db_i_sum += db_i
            dW_g_sum += dW_g
            db_g_sum += db_g
            dW_o_sum += dW_o
            db_o_sum += db_o
        
        # Ağırlıkları güncelle
        self.W_out -= self.lr * dW_out
        self.b_out -= self.lr * db_out
        self.W_f -= self.lr * dW_f_sum
        self.b_f -= self.lr * db_f_sum
        self.W_i -= self.lr * dW_i_sum
        self.b_i -= self.lr * db_i_sum
        self.W_g -= self.lr * dW_g_sum
        self.b_g -= self.lr * db_g_sum
        self.W_o -= self.lr * dW_o_sum
        self.b_o -= self.lr * db_o_sum
    
    def predict(self, x_sequence: np.ndarray) -> np.ndarray:
        """ 
        Tek bir sequence için Q-değerlerini tahmin eder.
        
        Args:
            x_sequence: Tek sequence (sequence_length, input_dim)
        
        Returns:
            Q-değerleri (out_dim,)
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if x_sequence.ndim != 2:
            raise ValueError(f"Input must be 2D (sequence_length, input_dim), got shape {x_sequence.shape}")
        q, _ = self.forward(x_sequence[None, :, :])
        return q[0]
    
    def copy_from(self, other: 'LSTM') -> None:
        """ 
        Başka bir LSTM'in ağırlıklarını bu ağa kopyalar (Target Network için).
        
        Args:
            other: Kopyalanacak LSTM nesnesi
        
        Raises:
            ValueError: Ağ yapıları uyumsuzsa
        """
        if (self.input_dim != other.input_dim or 
            self.hidden_dim != other.hidden_dim or 
            self.seq_len != other.seq_len or 
            self.out_dim != other.out_dim):
            raise ValueError(f"Network architectures must match: policy (input={self.input_dim}, hidden={self.hidden_dim}, seq={self.seq_len}, out={self.out_dim}) != other (input={other.input_dim}, hidden={other.hidden_dim}, seq={other.seq_len}, out={other.out_dim})")
        
        self.W_f = other.W_f.copy()
        self.b_f = other.b_f.copy()
        self.W_i = other.W_i.copy()
        self.b_i = other.b_i.copy()
        self.W_g = other.W_g.copy()
        self.b_g = other.b_g.copy()
        self.W_o = other.W_o.copy()
        self.b_o = other.b_o.copy()
        self.W_out = other.W_out.copy()
        self.b_out = other.b_out.copy()

# ---------------------------------------------------------------------
# 5. "HAFIZA": DENEYİM TEKRARI TAMPONU (REPLAY BUFFER)
# Bu sefer durumlar sequence formatında (son N adım).
# ---------------------------------------------------------------------
class SequenceBuffer:
    """
    Sequence'leri saklayan replay buffer.
    
    Normal replay buffer'dan farklı olarak, bu buffer sequence'leri saklar.
    Her deneyim, son N adımın durumlarını içeren bir sequence'dir.
    """
    def __init__(self, capacity: int = 5000, sequence_length: int = 10) -> None:
        """
        SequenceBuffer'ı başlatır.
        
        Args:
            capacity: Buffer kapasitesi (pozitif tam sayı)
            sequence_length: Sequence uzunluğu (pozitif tam sayı)
        
        Raises:
            ValueError: Geçersiz parametre değerleri
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if sequence_length <= 0:
            raise ValueError(f"Sequence length must be positive, got {sequence_length}")
        
        self.capacity = capacity
        self.seq_len = sequence_length
        self.buf: List[Tuple] = []
        self.pos = 0
    
    def push(
        self, 
        sequence: np.ndarray, 
        a: int, 
        r: float, 
        next_sequence: np.ndarray, 
        d: float
    ) -> None:
        """
        Bir sequence deneyimini hafızaya ekler.
        
        Args:
            sequence: (sequence_length, input_dim) boyutunda durum sequence'i
            a: Eylem
            r: Ödül
            next_sequence: Sonraki durum sequence'i
            d: Bitti mi? (done flag, 0.0 veya 1.0)
        """
        data = (sequence, a, r, next_sequence, d)
        if len(self.buf) < self.capacity:
            self.buf.append(data)
        else:
            self.buf[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Hafızadan rastgele bir batch sequence çeker.
        
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

def render_policy(env: GridWorld, net: LSTM, sequence_length: int = 10) -> None:
    """ 
    Öğrenilen politikayı ekrana çizer.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş LSTM ağı
        sequence_length: Sequence uzunluğu (pozitif tam sayı)
    
    Raises:
        ValueError: Geçersiz parametre değerleri
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    grid = []
    # Geçmiş durumları takip etmek için bir sequence oluştur
    # Her hücre için, o hücreyi son durum olarak kullanarak sequence oluştur
    past_states = [(0,0)] * (sequence_length - 1)  # Geçmiş durumlar (örnek)
    
    for y in range(env.h):
        row = []
        for x in range(env.w):
            if (x, y) == env.goal:
                row.append("G")
                continue
            
            # Sequence oluştur: geçmiş durumlar + mevcut durum
            seq = past_states + [(x, y)]
            seq_vectors = np.array([to_state_vector(s, env.w, env.h) for s in seq])
            
            a = int(np.argmax(net.predict(seq_vectors)))
            row.append(ACTIONS[a])
        grid.append(row)
    print("\n(LSTM-DQN) Öğrenilen Politika:")
    for r in grid: print(" ".join(r))

def evaluate(
    env: GridWorld, 
    net: LSTM, 
    episodes: int = 5, 
    max_steps: int = 100, 
    sequence_length: int = 10
) -> List[float]:
    """ 
    Eğitilmiş ajanın performansını test eder.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş LSTM ağı
        episodes: Test bölüm sayısı (pozitif tam sayı)
        max_steps: Her bölüm için maksimum adım sayısı (pozitif tam sayı)
        sequence_length: Sequence uzunluğu (pozitif tam sayı)
    
    Returns:
        Her bölüm için toplam ödül listesi
    
    Raises:
        ValueError: Geçersiz parametre değerleri
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    
    out: List[float] = []
    for _ in range(episodes):
        s = env.reset()
        R = 0.0
        sequence = []
        
        for _ in range(max_steps):
            # Sequence'e durum ekle
            s_vec = to_state_vector(s, env.w, env.h)
            sequence.append(s_vec)
            
            # Sequence'i doldur (ilk adımlarda tekrar et)
            if len(sequence) < sequence_length:
                seq_array = (sequence * (sequence_length // len(sequence) + 1))[:sequence_length]
            else:
                seq_array = sequence[-sequence_length:]
            
            seq_array = np.array(seq_array)
            
            # Q-değerlerini hesapla ve en iyi eylemi seç
            a = int(np.argmax(net.predict(seq_array)))
            s, r, done, _ = env.step(a)
            R += r
            if done: break
        out.append(R)
    return out

# ---------------------------------------------------------------------
# 3. ANA EĞİTİM FONKSİYONU
# Tüm LSTM-DQN öğrenme sürecini yönetir.
# ---------------------------------------------------------------------
def train_lstm_dqn(
    episodes=900, max_steps=200, gamma=0.99,
    eps_start=1.0, eps_min=0.01, eps_decay=0.995,
    lr=1e-3, batch=64, buf_cap=5000, start_after=400,
    target_every=200, use_target=True,
    sequence_length=10, hidden_dim=64, seed=123
):
    """
    LSTM-DQN eğitim fonksiyonu.
    
    MLP-DQN ve CNN-DQN'den farklı olarak, burada durumlar sequence
    formatında saklanır. Her deneyim, son N adımın durumlarını içerir.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = GridWorld()
    
    # LSTM ağları
    policy = LSTM(input_dim=2, hidden_dim=hidden_dim, sequence_length=sequence_length,
                  out_dim=4, lr=lr, seed=seed)
    target = LSTM(input_dim=2, hidden_dim=hidden_dim, sequence_length=sequence_length,
                  out_dim=4, lr=lr, seed=seed+1)
    target.copy_from(policy)
    
    rb = SequenceBuffer(buf_cap, sequence_length)
    eps = eps_start
    returns = []
    steps = 0
    
    # Ana eğitim döngüsü
    for ep in range(episodes):
        s = env.reset()
        epR = 0.0
        sequence = []  # Bu bölüm için durum sequence'i
        
        for t in range(max_steps):
            # Sequence'e durum ekle
            s_vec = to_state_vector(s, env.w, env.h)
            sequence.append(s_vec)
            
            # Sequence'i doldur (ilk adımlarda tekrar et)
            if len(sequence) < sequence_length:
                seq_array = (sequence * (sequence_length // len(sequence) + 1))[:sequence_length]
            else:
                seq_array = sequence[-sequence_length:]
            
            seq_array = np.array(seq_array)
            
            # Eylem seçimi
            q = policy.predict(seq_array)
            a = epsilon_greedy(q, eps)
            
            # Eylemi gerçekleştir
            s2, r, done, _ = env.step(a)
            s2_vec = to_state_vector(s2, env.w, env.h)
            
            # Sonraki sequence'i oluştur
            next_sequence = sequence + [s2_vec]
            if len(next_sequence) < sequence_length:
                next_seq_array = (next_sequence * (sequence_length // len(next_sequence) + 1))[:sequence_length]
            else:
                next_seq_array = next_sequence[-sequence_length:]
            
            next_seq_array = np.array(next_seq_array)
            
            # Deneyimi hafızaya kaydet
            rb.push(seq_array, a, r, next_seq_array, float(done))
            epR += r
            s = s2
            sequence = next_sequence  # Sequence'i güncelle
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
                policy.backward(S, cache, dq)
            
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
    # LSTM-DQN eğitimi
    env, policy, target, returns = train_lstm_dqn()
    # Öğrenilen politikayı göster
    render_policy(env, policy)
    # Performans değerlendirmesi
    print("Değerlendirme (sadece en iyi eylemlerle):", evaluate(env, policy, 10))

