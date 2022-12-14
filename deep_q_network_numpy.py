
# -*- coding: utf-8 -*-
"""
NumPy ile Sıfırdan Deep Q-Learning (DQN) (GridWorld)
---------------------------------------------------------
- Bu dosya, Tabular Q-Learning'in bir üst seviyesi olan DQN'i uygular.
- Q-tablosu yerine, durumları girdi olarak alıp her eylem için Q-değerlerini
  tahmin eden bir sinir ağı (MLP) kullanılır.
- Bu sayede, çok büyük veya sürekli durum uzaylarına sahip problemlerin
  çözümü mümkün hale gelir.
- Anahtar Konseptler:
    1. Fonksiyon Yaklaşımı: Q-değerlerini bir sinir ağı ile tahmin etme.
    2. Deneyim Tekrarı (Experience Replay): Öğrenme verimliliğini ve
       stabilitesini artırmak için geçmiş deneyimleri bir hafızada (buffer)
       saklayıp rastgele örneklemlerle ağı eğitme.
    3. Hedef Ağ (Target Network): Eğitimi stabilize etmek için periyodik
       olarak güncellenen ikinci bir sinir ağı kullanma.
Bu dosyayı tek başına çalıştırabilirsiniz.
"""
# ---------------------------------------------------------------------
# OKUMA SIRASI TAVSİYESİ
# ---------------------------------------------------------------------
# 1. `GridWorld` Sınıfı: Ajanın içinde yaşadığı ortam.
# 2. `if __name__ == "__main__"` Bloğu: Kodun ana akışı.
# 3. `train_dqn` Fonksiyonu: DQN eğitim sürecini yöneten ana döngü.
# 4. `MLP` Sınıfı: Q-değerlerini tahmin eden "beyin" (sinir ağı).
# 5. `ReplayBuffer` Sınıfı: Ajanın deneyimlerini sakladığı "hafıza".
# 6. Diğer yardımcı fonksiyonlar (`to_state_vector`, `epsilon_greedy` vb.).
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, List, Dict, Any
import numpy as np
import random

# Ortak modüllerden import
from common import GridWorld, to_state_vector, epsilon_greedy

# ---------------------------------------------------------------------
# 4. "BEYİN": ÇOK KATMANLI ALGILAYICI (MLP)
# Q-değerlerini tahmin etmek için kullanılacak sinir ağı.
# Bu sınıf, bir durum (state) verildiğinde, her bir olası eylem için
# beklenen gelecekteki ödülü (Q-değerini) tahmin etmeyi öğrenir.
# Kısacası, ajanın karar verme mekanizmasıdır.
# ---------------------------------------------------------------------
class MLP:
    """ Basit bir Çok Katmanlı Algılayıcı (Multi-Layer Perceptron). """
    def __init__(
        self, 
        in_dim: int = 2, 
        h1: int = 64, 
        h2: int = 64, 
        out_dim: int = 4, 
        lr: float = 1e-3, 
        seed: int = 42
    ) -> None:
        """
        MLP sinir ağını başlatır.
        
        Args:
            in_dim: Girdi boyutu (durum vektörü boyutu)
            h1: İlk gizli katman nöron sayısı
            h2: İkinci gizli katman nöron sayısı
            out_dim: Çıktı boyutu (eylem sayısı)
            lr: Öğrenme oranı
            seed: Rastgelelik tohumu
        
        Raises:
            ValueError: Geçersiz parametre değerleri
        """
        if in_dim <= 0 or h1 <= 0 or h2 <= 0 or out_dim <= 0:
            raise ValueError(f"All dimensions must be positive, got in_dim={in_dim}, h1={h1}, h2={h2}, out_dim={out_dim}")
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        
        # Ağırlıkları (weights) ve sapmaları (biases) ilklendir.
        # --- Ağırlıkların Başlatılması (He Initialization) ---
        # Neden rastgele sayılar `* np.sqrt(2.0/in_dim)` ile çarpılıyor?
        # Bu, "He initializaiton" olarak bilinen bir tekniktir. Özellikle ReLU
        # aktivasyon fonksiyonu kullanan ağlarda, öğrenme sürecinin başında
        # gradyanların çok küçülmesini (vanishing) veya çok büyümesini (exploding)
        # engeller. Bu sayede ağ daha stabil ve hızlı öğrenir. `in_dim` bir
        # önceki katmandaki nöron sayısıdır.
        # Matematiksel Sezgi: Bir katmanın çıktısının varyansının (verinin yayılımını)
        # girdisinin varyansına yakın tutmak hedeflenir. He yöntemi, bunun için
        # ağırlıkların varyansının `2 / girdi_sayısı` olması gerektiğini gösterir.
        # `rng.standard_normal` varyansı 1 olan sayılar ürettiği için, biz bu sayıları
        # standart sapma olan `np.sqrt(2.0 / in_dim)` ile çarparak istediğimiz
        # varyansı elde ederiz.
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((in_dim,h1)).astype(np.float32)*np.sqrt(2.0/in_dim)
        # --- Sapmaların Başlatılması ---
        # Sapmaları (bias) genellikle sıfır olarak başlatmak standart bir yaklaşımdır.
        # Ağırlıklar zaten düzgün bir şekilde başlatıldığı için, sapmaların nötr bir
        # başlangıç yapması en güvenli yoldur. Ağ, eğitim sırasında bu değerleri
        # öğrenerek kendisi ayarlayacaktır.
        self.b1 = np.zeros((h1,), np.float32)
        self.W2 = rng.standard_normal((h1,h2)).astype(np.float32)*np.sqrt(2.0/h1)
        self.b2 = np.zeros((h2,), np.float32)
        self.W3 = rng.standard_normal((h2,out_dim)).astype(np.float32)*np.sqrt(2.0/h2)
        self.b3 = np.zeros((out_dim,), np.float32)
        self.lr = lr # Öğrenme oranı (learning rate)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ ReLU (Rectified Linear Unit) aktivasyon fonksiyonu. """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        """ ReLU'nun türevi (geri yayılım için gerekli). """
        return (x > 0).astype(np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """
        İleri Yayılım (Forward Pass): Girdiyi (durum vektörü) alır ve
        Q-değerlerini (çıktı) hesaplar.
        Matematiksel olarak her katmanda `Aktivasyon(girdi @ Ağırlık + Sapma)`
        işlemi yapılır. Geri yayılım (backward) için ara değerleri
        (cache) de saklarız çünkü bu değerler gradyan hesaplamasında
        tekrar kullanılacaktır. Bu, tekrar hesaplama yapmaktan kurtarır.
        
        Args:
            x: Girdi batch'i (batch_size, in_dim)
        
        Returns:
            Tuple containing:
                - q: Q-değerleri (batch_size, out_dim)
                - cache: Geri yayılım için ara değerler
        
        Raises:
            ValueError: Girdi boyutu beklenen boyutla eşleşmiyorsa
        """
        if x.ndim != 2 or x.shape[1] != self.W1.shape[0]:
            raise ValueError(f"Input shape mismatch: expected (batch, {self.W1.shape[0]}), got {x.shape}")
        # Katman 1
        z1 = x @ self.W1 + self.b1; a1 = self.relu(z1)
        # Katman 2
        z2 = a1 @ self.W2 + self.b2; a2 = self.relu(z2)
        # Çıkış katmanı (Q-değerleri)
        q = a2 @ self.W3 + self.b3
        return q, (x,z1,a1,z2,a2)

    def backward(self, cache: Tuple, dq: np.ndarray) -> None:
        """
        Geri Yayılım (Backward Pass): Hata gradyanını (dq) kullanarak
        ağın ağırlıklarını günceller (öğrenme anı). Bu, sinir ağının
        "öğrendiği" yerdir. Zincir kuralı (chain rule) kullanılarak,
        çıktıdaki hatadan geriye doğru her bir ağırlığın bu hataya olan
        katkısı (gradyanı) hesaplanır ve ağırlıklar bu gradyanın tersi
        yönünde güncellenir (Gradient Descent).
        
        Args:
            cache: Forward pass'ten gelen ara değerler
            dq: Çıktı katmanı gradyanları (batch_size, out_dim)
        
        Raises:
            ValueError: Gradyan boyutu beklenen boyutla eşleşmiyorsa
        """
        if dq.shape[1] != self.W3.shape[1]:
            raise ValueError(f"Gradient shape mismatch: expected (batch, {self.W3.shape[1]}), got {dq.shape}")
        x,z1,a1,z2,a2 = cache
        B = x.shape[0] # Batch boyutu (aynı anda işlenen örnek sayısı)
        # Çıkış katmanı gradyanları
        # dW3: a2'nin transpozu ile dq'nun matris çarpımı. Bu, 3. katman ağırlıklarının gradyanını verir.
        dW3 = a2.T @ dq / B; db3 = dq.mean(axis=0)
        # 2. gizli katman gradyanları
        da2 = dq @ self.W3.T; dz2 = da2 * self.relu_grad(z2)
        dW2 = a1.T @ dz2 / B; db2 = dz2.mean(axis=0)
        # 1. gizli katman gradyanları
        da1 = dz2 @ self.W2.T; dz1 = da1 * self.relu_grad(z1)
        dW1 = x.T @ dz1 / B; db1 = dz1.mean(axis=0)
        # Ağırlıkları ve sapmaları güncelle (Gradient Descent adımı)
        # Yeni Ağırlık = Eski Ağırlık - ÖğrenmeOranı * Gradyan
        self.W3 -= self.lr*dW3; self.b3 -= self.lr*db3
        self.W2 -= self.lr*dW2; self.b2 -= self.lr*db2
        self.W1 -= self.lr*dW1; self.b1 -= self.lr*db1

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
        if x_single.ndim != 1 or x_single.shape[0] != self.W1.shape[0]:
            raise ValueError(f"Input shape mismatch: expected ({self.W1.shape[0]},), got {x_single.shape}")
        # forward metodu bir batch beklediği için girdiyi genişletiyoruz (None,:).
        q, _ = self.forward(x_single[None, :])
        return q[0]

    def copy_from(self, other: 'MLP') -> None:
        """ 
        Başka bir MLP'nin ağırlıklarını bu ağa kopyalar (Target Network için).
        
        Args:
            other: Kopyalanacak MLP nesnesi
        
        Raises:
            ValueError: Ağ yapıları uyumsuzsa
        """
        if (self.W1.shape != other.W1.shape or 
            self.W2.shape != other.W2.shape or 
            self.W3.shape != other.W3.shape):
            raise ValueError(f"Network architectures must match: policy {self.W1.shape, self.W2.shape, self.W3.shape} != other {other.W1.shape, other.W2.shape, other.W3.shape}")
        
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

# ---------------------------------------------------------------------
# 5. "HAFIZA": DENEYİM TEKRARI TAMPONU (REPLAY BUFFER)
# Ajanın (durum, eylem, ödül, yeni_durum) deneyimlerini saklar.
# Neden böyle bir yapıya ihtiyacımız var?
# 1. Korelasyonu Kırmak: Ajanın ardışık adımları birbirine çok benzer (yüksek
#    korelasyonludur). Sinir ağını sürekli bu benzer verilerle eğitmek,
#    öğrenmeyi istikrarsızlaştırır. Rastgele deneyimler seçerek bu
#    korelasyonu kırarız ve ağın daha genel bir politika öğrenmesini sağlarız.
# 2. Veri Verimliliği: Ajanın yaşadığı her bir deneyim (adım), hafızaya
#    atılır ve birden çok kez eğitim için kullanılabilir. Bu, özellikle
#    gerçek dünyada deneyim toplamanın maliyetli olduğu durumlarda çok
#    önemlidir.
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
        Bir denetimi hafızaya ekler. Hafıza doluysa en eskisinin üzerine yazar.
        Bu yapıya "dairesel tampon" (circular buffer) denir. `pos` değişkeni
        bir sonraki verinin nereye yazılacağını gösterir. Kapasiteye ulaşıldığında
        `pos` sıfırlanır ve en eski verinin üzerine yazmaya başlar.
        
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
        Hafızadan rastgele bir `batch` boyutunda deneyim örneği çeker.
        Bu, DQN eğitiminin en kritik adımlarından biridir. Ağı eğitmek için
        her zaman bu rastgele çekilmiş mini-batch'i kullanırız.
        `zip(*...)` ve `np.stack` gibi operasyonlar, listeler halindeki
        veriyi sinir ağının işleyebileceği NumPy dizilerine (tensörlere)
        dönüştürmek için kullanılır.
        
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
ACTIONS = {0:"↑",1:"↓",2:"←",3:"→"}

def render_policy(env: GridWorld, net: MLP) -> None:
    """ 
    Öğrenilen politikayı (her durumda en iyi eylemi) ekrana çizer.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş MLP ağı
    """
    grid: List[List[str]] = []
    for y in range(env.h):
        row: List[str] = []
        for x in range(env.w):
            if (x, y) == env.goal:
                row.append("G")
                continue
            s = to_state_vector((x, y), env.w, env.h)
            a = int(np.argmax(net.predict(s)))
            if a not in ACTIONS:
                raise ValueError(f"Invalid action {a} found in policy at position ({x}, {y})")
            row.append(ACTIONS[a])
        grid.append(row)
    print("\n(DQN) Öğrenilen Politika:")
    for r in grid:
        print(" ".join(r))

def evaluate(
    env: GridWorld, 
    net: MLP, 
    episodes: int = 5, 
    max_steps: int = 100
) -> List[float]:
    """ 
    Eğitilmiş ajanın performansını (keşif olmadan) test eder.
    
    Args:
        env: GridWorld ortamı
        net: Eğitilmiş MLP ağı
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
            # Sadece en iyi bilinen eylemi seç (argmax).
            a = int(np.argmax(net.predict(to_state_vector(s, env.w, env.h))))
            s, r, done, _ = env.step(a)
            R += r
            if done:
                break
        out.append(R)
    return out

# ---------------------------------------------------------------------
# 3. ANA EĞİTİM FONKSİYONU
# Tüm DQN öğrenme sürecini yönetir.
# ---------------------------------------------------------------------
def train_dqn(
    # ----- Hiperparametreler -----
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
) -> Tuple[GridWorld, MLP, MLP, List[float]]:
    """
    DQN algoritması ile GridWorld problemini çözer.
    
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
        Tuple containing:
            - env: GridWorld ortamı
            - policy: Eğitilmiş politika ağı
            - target: Hedef ağ
            - returns: Her bölüm için toplam ödül listesi
    
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

    # 1. Ana Ağ (Policy Network): Eylemleri seçmek ve eğitilmek için kullanılır.
    policy = MLP(2,64,64,4,lr,seed)
    # 2. Hedef Ağ (Target Network): TD hedefini hesaplamak için kullanılır.
    #    Bu ağın yavaş güncellenmesi, eğitimi stabilize eder.
    target = MLP(2,64,64,4,lr,seed+1); target.copy_from(policy)
    # 3. Deneyim Tekrarı Hafızası
    rb = ReplayBuffer(buf_cap)
    eps = eps_start; returns=[]; steps=0

    # Ana eğitim döngüsü
    for ep in range(episodes):
        s = env.reset(); epR = 0.0
        for t in range(max_steps):
            # 1. Eylem Seçimi
            q = policy.predict(to_state_vector(s, env.w, env.h))
            a = epsilon_greedy(q, eps)

            # 2. Eylemi Gerçekleştirme ve Deneyimi Saklama
            s2, r, done, _ = env.step(a)
            # (s, a, r, s', done) beşlisini hafızaya kaydet.
            rb.push(to_state_vector(s, env.w, env.h), a, r, to_state_vector(s2, env.w, env.h), float(done))
            epR += r; s = s2; steps += 1

            # 3. Ağı Eğitme (ÖĞRENME ANI)
            # Yeterli deneyim toplandıktan sonra eğitime başla.
            if len(rb) >= max(batch, start_after):
                # 3.1. Hafızadan rastgele bir batch deneyim al.
                S,A,R,S2,D = rb.sample(batch)

                # 3.2. Ana ağ ile mevcut durumlar için Q-değerlerini hesapla.
                Qs, cache = policy.forward(S)

                # 3.3. Hedef ağı ile sonraki durumlar için Q-değerlerini hesapla.
                if use_target: Qs2,_ = target.forward(S2)
                else: Qs2,_ = policy.forward(S2) # Hedef ağsız denemek için

                # 3.4. TD Hedefini Hesapla (Bellman Denklemi)
                # y = R                      (eğer bölüm bittiyse)
                # y = R + γ * max(Q_target(s')) (eğer bölüm devam ediyorsa)
                y = R + gamma*(1.0-D)*np.max(Qs2,axis=1)

                # 3.5. Gradyanı Hesapla (Kayıp Fonksiyonunun Türevi)
                # Kayıp (Loss) = (y - Q_policy(s,a))^2 (MSE)
                # Bu kaybın Q_policy'ye göre türevi, dq'yu oluşturur.
                dq = np.zeros_like(Qs)
                idx = np.arange(batch)
                dq[idx, A] = (Qs[idx, A] - y) # Sadece seçilen eylemlerin gradyanı hesaplanır.

                # 3.6. Ana ağı geri yayılım ile güncelle.
                policy.backward(cache, dq)

            if done: break
            # 4. Hedef Ağı Güncelleme
            # Belirli aralıklarla ana ağın ağırlıklarını hedef ağa kopyala.
            if use_target and steps % target_every == 0: target.copy_from(policy)

        returns.append(epR)
        # Epsilon'u azaltarak keşfi zamanla azalt.
        eps = max(eps_min, eps*eps_decay)

    return env, policy, target, returns

# ---------------------------------------------------------------------
# 2. ANA AKIŞ (ENTRY POINT)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # `train_dqn` fonksiyonunu çağırarak ajanı eğit.
    env, policy, target, returns = train_dqn()
    # Öğrenilen politikayı ekrana yazdır.
    render_policy(env, policy)
    # Eğitilmiş ajanın performansını 10 bölüm boyunca test et.
    print("Değerlendirme (sadece en iyi eylemlerle):", evaluate(env, policy, 10))
