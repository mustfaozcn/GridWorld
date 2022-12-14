
# -*- coding: utf-8 -*-
"""
En basit haliyle Tabular Q-Learning (GridWorld)
------------------------------------------------
- Küçük, deterministik bir 5x5 GridWorld
- Her adımda -1 ödül, hedefe ulaşınca +10 ve bölüm biter
- Dört eylem: 0=Yukarı, 1=Aşağı, 2=Sol, 3=Sağ
- ε-greedy politika, sabit / kademeli azalan keşif
Bu dosyayı tek başına çalıştırabilirsiniz.
"""
# ---------------------------------------------------------------------
# GridWorld Koordinat Sistemi Görselleştirmesi (x, y)
# ---------------------------------------------------------------------
#
#     sütun 0    sütun 1    sütun 2    sütun 3    sütun 4
#  +----------+----------+----------+----------+----------+
#  |          |          |          |          |          | satır 0
#  |  (0,0) S |  (1,0)   |  (2,0)   |  (3,0)   |  (4,0)   |
#  |          |          |          |          |          |
#  +----------+----------+----------+----------+----------+
#  |          |          |          |          |          | satır 1
#  |  (0,1)   |  (1,1)   |  (2,1)   |  (3,1)   |  (4,1)   |
#  |          |          |          |          |          |
#  +----------+----------+----------+----------+----------+
#  |          |          |          |          |          | satır 2
#  |  (0,2)   |  (1,2)   |  (2,2)   |  (3,2)   |  (4,2)   |
#  |          |          |          |          |          |
#  +----------+----------+----------+----------+----------+
#  |          |          |          |          |          | satır 3
#  |  (0,3)   |  (1,3)   |  (2,3)   |  (3,3)   |  (4,3)   |
#  |          |          |          |          |          |
#  +----------+----------+----------+----------+----------+
#  |          |          |          |          |          | satır 4
#  |  (0,4)   |  (1,4)   |  (2,4)   |  (3,4)   |  (4,4) G |
#  |          |          |          |          |          |
#  +----------+----------+----------+----------+----------+
#
# S: Başlangıç Noktası (start)
# G: Hedef Noktası (goal)
# Q-tablosu indekslemesi Q[y, x, eylem] şeklindedir. Örneğin (3,2) konumundaki
# 'yukarı' eyleminin değeri Q[2, 3, 0] olur.
# ---------------------------------------------------------------------

# ---> OKUMAYA BAŞLAMADAN ÖNCE:
# Bu kodun nasıl çalıştığını anlamak için aşağıdaki sırayla okumanızı tavsiye ederim:
# 1. `GridWorld` Sınıfı: Ajanın içinde yaşadığı ve kuralları belirleyen dünyayı anlamak için.
# 2. `if __name__ == "__main__"` Bloğu: Kodun ana akışını, yani hangi fonksiyonların hangi sırayla çağrıldığını görmek için.
# 3. `train` Fonksiyonu: Genel eğitim sürecini yöneten ana döngüyü anlamak için.
# 4. `run_episode` Fonksiyonu: Q-Learning'in kalbi olan, tek bir eğitim bölümünde Q-tablosunun nasıl güncellendiğini görmek için.
# 5. Diğer yardımcı fonksiyonlar (`epsilon_greedy`, `render_policy` vb.) `run_episode` içindeki adımları detaylandırır.

# Kütüphaneleri içe aktaralım
from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np  # Sayısal hesaplamalar için güçlü bir kütüphane (Q-tablosu, matematiksel işlemler vb.)
import random       # Rastgele seçimler yapmak için (örneğin epsilon-greedy stratejisi için)

# Ortak modüllerden import
from common import GridWorld
from common.utils import epsilon_greedy as epsilon_greedy_base

# ---------------------------------------------------------------------
# 1. ADIM: ORTAMI ANLAMAK
# GridWorld artık common modülünden import ediliyor.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# Bu fonksiyonlar, ana eğitim döngüsü içinde belirli görevleri yerine getirmek için kullanılır.
# ---------------------------------------------------------------------
ACTIONS = {
    # Eylem numaralarını okunabilir oklara çeviren bir sözlük.
    # Politikayı ekrana yazdırırken kullanılır.
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}

def epsilon_greedy(Q: np.ndarray, state: Tuple[int, int], epsilon: float) -> int:
    """
    ---> ANLAMAK İÇİN: Bu fonksiyon, `run_episode` içinde ajanın bir sonraki adımda hangi eylemi
    yapacağına karar verirken kullanılır.

    ε-greedy (epsilon-açgözlü) eylem seçimi stratejisi.
    Bu strateji, ajanın keşfetme (exploration) ve kullanma (exploitation) arasında
    bir denge kurmasını sağlar.

    - Keşfetme: Ajan, en iyi olduğunu düşündüğü eylem yerine rastgele bir eylem seçer.
      Böylece daha önce denemediği yolları öğrenir.
    - Kullanma: Ajan, o ana kadar öğrendiği en iyi eylemi (en yüksek Q-değerine sahip eylemi) seçer.
    
    Bu fonksiyon, Q-tablosundan ilgili q vektörünü çıkarır ve ortak epsilon_greedy_base
    fonksiyonunu çağırır.
    
    Args:
        Q: Q-tablosu (yükseklik, genişlik, 4) boyutunda numpy array
        state: Mevcut durum koordinatları (x, y)
        epsilon: Keşfetme olasılığı (0.0 ile 1.0 arası)
    
    Returns:
        Seçilen eylem indeksi (0-3 arası)
    
    Raises:
        ValueError: Q-tablosunun boyutu uygun değilse veya state geçersizse
        IndexError: State koordinatları Q-tablosunun sınırları dışındaysa
    """
    if Q.ndim != 3 or Q.shape[2] != 4:
        raise ValueError(f"Q-table must have shape (h, w, 4), got {Q.shape}")
    
    x, y = state
    if not isinstance(state, (tuple, list)) or len(state) != 2:
        raise ValueError(f"State must be a tuple/list of 2 integers, got {state}")
    if not (0 <= x < Q.shape[1] and 0 <= y < Q.shape[0]):
        raise IndexError(f"State coordinates ({x}, {y}) out of bounds for Q-table shape {Q.shape[:2]}")
    
    # Q-tablosunda (y, x) konumu için 4 eylemin Q-değerlerini al
    # DİKKAT: Q-tablosu [y, x, eylem] sırasında indekslenir, çünkü NumPy dizileri
    # genellikle (satır, sütun) yani (yükseklik, genişlik) olarak düşünülür.
    q_values = Q[y, x, :]  # 4 elemanlı vektör: [Q(yukarı), Q(aşağı), Q(sol), Q(sağ)]
    # Ortak epsilon_greedy fonksiyonunu çağır
    return epsilon_greedy_base(q_values, epsilon)

def render_policy(Q: np.ndarray, env: GridWorld) -> None:
    """ 
    Öğrenilen politikayı (en iyi eylem haritasını) yön oklarıyla ekrana yazdırır.
    
    Args:
        Q: Q-tablosu (yükseklik, genişlik, 4) boyutunda numpy array
        env: GridWorld ortamı
    
    Raises:
        ValueError: Q-tablosunun boyutu ortam boyutlarıyla uyumsuzsa
    """
    if Q.ndim != 3 or Q.shape[0] != env.h or Q.shape[1] != env.w or Q.shape[2] != 4:
        raise ValueError(f"Q-table shape {Q.shape} does not match environment size ({env.h}, {env.w}, 4)")
    
    grid = []
    for y in range(env.h):
        row = []
        for x in range(env.w):
            if (x, y) == env.goal:
                # Eğer mevcut (x, y) koordinatı hedef ise, 'G' harfi koy.
                row.append("G")
            else:
                # Değilse, o koordinattaki en iyi eylemi bul.
                a = int(np.argmax(Q[y, x, :]))
                # Eylem numarasını ACTIONS sözlüğüyle oka çevir (örn: 3 -> "→").
                if a not in ACTIONS:
                    raise ValueError(f"Invalid action {a} found in Q-table at position ({x}, {y})")
                row.append(ACTIONS[a])
        grid.append(row)

    print("\nÖğrenilen Politika (G=Hedef):")
    for row in grid:
        # Her satırı "→ ↓ ↓ → G" gibi bir formatta yazdırır.
        print(" ".join(row))

def run_episode(
    env: GridWorld, 
    Q: np.ndarray, 
    alpha: float, 
    gamma: float, 
    epsilon: float, 
    max_steps: int = 200
) -> float:
    """
    ---> 4. ADIM: Q-LEARNING'İN KALBİ
    Bu fonksiyon, Q-öğrenme algoritmasının özünü içerir. Ajanın bir bölüm boyunca
    deneyim kazanmasını ve bu deneyimlerle Q-tablosunu (beynini) güncellemesini sağlar.
    `train` fonksiyonundaki ana döngü tarafından tekrar tekrar çağrılır.

    Tek bir eğitim bölümünü (episode) baştan sona çalıştırır.
    Ajan başlangıç noktasından hedefe ulaşana veya maksimum adım sayısına gelene kadar
    hareket eder ve bu sırada Q-tablosunu günceller.
    
    Args:
        env: GridWorld ortamı
        Q: Q-tablosu (yükseklik, genişlik, 4) boyutunda numpy array
        alpha: Öğrenme oranı (0.0 ile 1.0 arası)
        gamma: İndirgeme faktörü (0.0 ile 1.0 arası)
        epsilon: Keşfetme olasılığı (0.0 ile 1.0 arası)
        max_steps: Maksimum adım sayısı (pozitif tam sayı)
    
    Returns:
        Toplam ödül (float)
    
    Raises:
        ValueError: Geçersiz hiperparametre değerleri veya Q-tablosu boyutu uyumsuzsa
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"Gamma must be between 0.0 and 1.0, got {gamma}")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"Epsilon must be between 0.0 and 1.0, got {epsilon}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if Q.ndim != 3 or Q.shape[0] != env.h or Q.shape[1] != env.w or Q.shape[2] != 4:
        raise ValueError(f"Q-table shape {Q.shape} does not match environment size ({env.h}, {env.w}, 4)")
    
    s = env.reset()  # Ortamı sıfırla, ajanı başlangıç konumuna (s) al.
    total_reward = 0.0
    for _ in range(max_steps): # Bölümün çok uzun sürmesini engellemek için bir sınır.
        # 1. Adım: Eylem Seçimi
        # Ajan mevcut durumunda ne yapacağına karar verir.
        # ---> Bu adımın detayları için `epsilon_greedy` fonksiyonuna bakın.
        a = epsilon_greedy(Q, s, epsilon)

        # 2. Adım: Eylemi Gerçekleştirme ve Sonuçları Gözlemleme
        # Ajan, seçtiği eylemi dünyada gerçekleştirir ve sonuçlarını (yeni durum, ödül) alır.
        # ---> Bu adımın detayları için `GridWorld` sınıfının `step` metoduna bakın.
        s_next, r, done, _ = env.step(a)

        # Koordinatları daha kolay kullanmak için ayıralım.
        x, y = s      # Önceki durumun koordinatları
        nx, ny = s_next # Yeni durumun koordinatları

        # 3. Adım: Q-Tablosunu Güncelleme (ÖĞRENME ANI)
        # Burası, ajanın deneyimlerinden bir şeyler öğrendiği yerdir.
        # Bellman denklemini kullanarak Q-değerini güncelleriz.
        # Formül: Q(s, a) = Q(s, a) + α * [ R + γ * max(Q(s', a')) - Q(s, a) ]
        # Anlamı: "Mevcut durumdaki eylemin değerini, elde edilen anlık ödül ve
        # bir sonraki durumdan elde edilebilecek en iyi gelecekteki ödülün birleşimiyle
        # güncelleyerek hedefe yaklaştır."

        # 3.1: TD Hedefi (Temporal Difference Target) - Olması gereken ideal değer
        # TD Hedefi = R + γ * max(Q(s', a'))
        # Bu, ajanın o adımda ulaşmayı hedeflediği "ideal" Q-değeridir.
        # r: anlık alınan ödül.
        # gamma * np.max(Q[ny, nx, :]): bir sonraki adımdaki en iyi potansiyelin bugünkü değeri.
        # (0.0 if done else 1.0) -> Eğer bölüm bittiyse (hedefe ulaşıldıysa), gelecek yoktur,
        # bu yüzden gelecek ödül beklentisi 0 olur.
        td_target = r + gamma * np.max(Q[ny, nx, :]) * (0.0 if done else 1.0)

        # 3.2: TD Hatası (Temporal Difference Error) - Beklenti ile gerçek arasındaki fark
        # TD Hatası = TD Hedefi - Mevcut Q-değeri
        # Bu, beklentimiz (td_target) ile mevcut tahminimiz (Q[y, x, a]) arasındaki farktır.
        # Hata ne kadar büyükse, öğrenme o kadar etkili olur.
        td_error = td_target - Q[y, x, a]

        # 3.3: Q-Değerini Güncelleme - Beyni (Q-tablosunu) GÜNCELLEME
        # Mevcut Q-değerine, öğrenme oranı (alpha) ile ölçeklendirilmiş hatayı ekleriz.
        # Bu, Q-değerini yavaş yavaş "doğru" değere yakınlaştırır.
        Q[y, x, a] += alpha * td_error

        # Döngünün bir sonraki adımı için hazırlık
        total_reward += r # Bu bölümdeki toplam ödülü takip et.
        s = s_next        # Mevcut durumu yeni durumla güncelle.

        if done:
            # Eğer hedefe ulaşıldıysa, bu bölümü bitir.
            break
    return total_reward # Bölüm sonunda elde edilen toplam ödülü döndür.

def evaluate(env: GridWorld, Q: np.ndarray, episodes: int = 5, max_steps: int = 100) -> List[float]:
    """
    Eğitilmiş politikayı değerlendirir.
    Bu fonksiyonda keşfetme (epsilon) yoktur, ajan her zaman en iyi bildiği
    (en yüksek Q-değerine sahip) eylemi seçer.
    
    Args:
        env: GridWorld ortamı
        Q: Q-tablosu (yükseklik, genişlik, 4) boyutunda numpy array
        episodes: Değerlendirme için çalıştırılacak bölüm sayısı (pozitif tam sayı)
        max_steps: Her bölüm için maksimum adım sayısı (pozitif tam sayı)
    
    Returns:
        Her bölüm için toplam ödül listesi
    
    Raises:
        ValueError: Geçersiz parametre değerleri veya Q-tablosu boyutu uyumsuzsa
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if Q.ndim != 3 or Q.shape[0] != env.h or Q.shape[1] != env.w or Q.shape[2] != 4:
        raise ValueError(f"Q-table shape {Q.shape} does not match environment size ({env.h}, {env.w}, 4)")
    
    rewards = []
    for _ in range(episodes):
        s = env.reset()
        ep_r = 0.0
        for _ in range(max_steps):
            x, y = s
            # Epsilon-greedy yerine doğrudan en iyi eylemi seç (np.argmax).
            a = int(np.argmax(Q[y, x, :]))
            s, r, done, _ = env.step(a)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
    return rewards

# ---------------------------------------------------------------------
# 3. ADIM: ANA EĞİTİM SÜRECİNİ ANLAMAK
# Bu fonksiyon, tüm öğrenme sürecini baştan sona yönetir.
# Gerekli hazırlıkları yapar ve `run_episode` fonksiyonunu defalarca çağırarak
# ajanın giderek daha akıllı hale gelmesini sağlar.
# ---------------------------------------------------------------------
def train(
    # ----- Hiperparametreler (Modelin nasıl öğreneceğini kontrol eden ayarlar) -----
    episodes: int = 800,         # Toplam eğitim bölümü sayısı. Ne kadar çok, o kadar iyi öğrenir (genellikle).
    alpha: float = 0.1,            # Öğrenme Oranı (Learning Rate): Yeni bilginin eski bilgiyi ne kadar ezeceğini belirler.
                                   # 0: Hiç öğrenme, 1: Sadece yeni bilgiyi dikkate al.
    gamma: float = 0.99,           # İndirgeme Faktörü (Discount Factor): Gelecekteki ödüllerin bugünkü değerini belirler.
                                   # 0: Sadece anlık ödülü önemse, ~1: Gelecekteki ödülleri çok önemse.
    epsilon_start: float = 1.0,    # Başlangıç Epsilon'u: Başta %100 rastgele hareket et (tam keşif).
    epsilon_min: float = 0.01,     # Minimum Epsilon: Keşfetmenin asla %1'in altına düşmemesini sağlar.
    epsilon_decay: float = 0.995,  # Epsilon Azalma Oranı: Her bölüm sonunda epsilon bu değerle çarpılır.
                                   # Bu, ajanın zamanla keşfetmeyi azaltıp öğrendiklerini kullanmaya başlamasını sağlar.
    optimistic_init: float = 0.0,  # İyimser Başlangıç Değeri: Q-tablosunun başlangıç değeri.
                                   # Yüksek bir değer (örn: 5.0) ajanı daha önce gitmediği yerleri denemeye teşvik eder.
    seed: int = 42                 # Rastgelelik Tohumu: Kodun her çalıştırmada aynı "rastgele" sonuçları üretmesini sağlar.
) -> Tuple[GridWorld, np.ndarray, List[float]]:
    """
    Q-Learning algoritması ile GridWorld problemini çözer.
    
    Args:
        episodes: Toplam eğitim bölümü sayısı (pozitif tam sayı)
        alpha: Öğrenme oranı (0.0 ile 1.0 arası)
        gamma: İndirgeme faktörü (0.0 ile 1.0 arası)
        epsilon_start: Başlangıç epsilon değeri (0.0 ile 1.0 arası)
        epsilon_min: Minimum epsilon değeri (0.0 ile epsilon_start arası)
        epsilon_decay: Epsilon azalma oranı (0.0 ile 1.0 arası)
        optimistic_init: Q-tablosu başlangıç değeri
        seed: Rastgelelik tohumu
    
    Returns:
        Tuple containing:
            - env: GridWorld ortamı
            - Q: Eğitilmiş Q-tablosu
            - returns: Her bölüm için toplam ödül listesi
    
    Raises:
        ValueError: Geçersiz hiperparametre değerleri
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be between 0.0 and 1.0, got {gamma}")
    if not 0.0 <= epsilon_start <= 1.0:
        raise ValueError(f"epsilon_start must be between 0.0 and 1.0, got {epsilon_start}")
    if not 0.0 <= epsilon_min <= epsilon_start:
        raise ValueError(f"epsilon_min must be between 0.0 and epsilon_start ({epsilon_start}), got {epsilon_min}")
    if not 0.0 <= epsilon_decay <= 1.0:
        raise ValueError(f"epsilon_decay must be between 0.0 and 1.0, got {epsilon_decay}")
    
    # Rastgelelik tohumlarını ayarlayarak deneylerin tekrarlanabilir olmasını sağlıyoruz.
    random.seed(seed)
    np.random.seed(seed)

    # Ortamı oluştur.
    env = GridWorld()
    # Q-tablosunu oluştur.
    # Boyut: (yükseklik, genişlik, eylem_sayısı) -> (5, 5, 4)
    # np.full ile tüm değerleri başlangıçta `optimistic_init` (varsayılan 0.0) ile doldururuz.
    Q = np.full((env.h, env.w, 4), fill_value=optimistic_init, dtype=np.float32)

    # Epsilon değerini takip etmek için bir değişken.
    eps = epsilon_start
    returns = []

    # Ana eğitim döngüsü: Ajanın tekrar tekrar deneme yaparak öğrenmesini sağlar.
    for ep in range(episodes):
        # Tek bir bölüm çalıştır ve Q-tablosunu güncelle.
        # ---> ÖĞRENMENİN GERÇEKLEŞTİĞİ YER OLAN `run_episode` fonksiyonuna bakın.
        G = run_episode(env, Q, alpha, gamma, eps)
        returns.append(G) # Bölümden gelen toplam ödülü kaydet.

        # Epsilon'u azalt (decay). Bu, ajanın zamanla daha az rastgele hareket edip,
        # öğrendiklerini daha fazla kullanmasını (exploitation) sağlar.
        # Örnek: eps = 1.0 -> 1.0 * 0.995 = 0.995
        #        eps = 0.995 -> 0.995 * 0.995 = 0.990025
        # Bu şekilde ajan zamanla daha az rastgele hareket eder.
        eps = max(epsilon_min, eps * epsilon_decay)

    # Eğitim bittiğinde öğrenilmiş ortamı, Q-tablosunu ve ödül listesini döndür.
    return env, Q, returns

if __name__ == "__main__":
    # ---> 2. ADIM: KODUN BAŞLANGIÇ NOKTASI
    # Bu blok, dosya doğrudan çalıştırıldığında yürütülür. Kodun ana akışı buradadır.
    # 1. `train` fonksiyonu çağrılarak ajan eğitilir.
    # 2. `render_policy` ile ajanın öğrendiği politika (en iyi yol haritası) çizdirilir.
    # 3. `evaluate` ile eğitilmiş ajanın performansı test edilir.

    # `train` fonksiyonunu belirlenen hiperparametrelerle çağır.
    # ---> Bu fonksiyonun detayları için yukarıdaki `train` fonksiyon tanımına gidin.
    env, Q, returns = train(
        episodes=800,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        optimistic_init=0.0,  # İsterseniz 5.0 yapıp "iyimser başlangıç" etkisini deneyin
        seed=42
    )

    # Eğitim sonunda öğrenilen en iyi politikayı ekrana yazdır.
    # ---> Bu fonksiyonun detayı için `render_policy` tanımına bakın.
    render_policy(Q, env)

    # Son olarak, öğrenilen politikayı keşfetme olmadan (sadece en iyi eylemleri kullanarak)
    # 10 defa çalıştırıp ne kadar başarılı olduğunu test et.
    # ---> Bu fonksiyonun detayı için `evaluate` tanımına bakın.
    eval_rewards = evaluate(env, Q, episodes=10)
    print("\nDeğerlendirme (sadece en iyi eylemlerle) bölüm ödülleri:", eval_rewards)
    print("Ortalama değerlendirme ödülü:", np.mean(eval_rewards))

    # Ek bir ipucu
    print("\nİpucu: 'optimistic_init=5.0' gibi bir değer, ajanın keşfetmesini artırır.")
