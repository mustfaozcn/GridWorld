# DQN Nasıl Çalışır?

---

## MLP Mimarisi

DQN, Multi-Layer Perceptron (MLP) kullanarak Q-değerlerini öğrenir. MLP, ajanın mevcut durumunu (konumu) girdi olarak alır ve her olası eylem için bir Q-değeri üretir.

### Girdi ve Çıktı Katmanları

- **Girdi Katmanı (2 Nöron):** Ajanın konumunu temsil eder
  - `x` koordinatı: Bir nöron
  - `y` koordinatı: Bir nöron
  - Bu 2 değer, GridWorld'deki ajanın pozisyonunu kodlar

- **Çıktı Katmanı (4 Nöron):** Her eylem için Q-değeri
  - Yukarı eylemi için Q-değeri
  - Aşağı eylemi için Q-değeri
  - Sol eylemi için Q-değeri
  - Sağ eylemi için Q-değeri
  - En yüksek Q-değerli eylem seçilir

### Gizli Katmanlar

Girdi ve çıktı arasındaki 64'er nöronluk iki gizli katman, ağın düşünme ve akıl yürütme sürecini gerçekleştirir. Bu katmanları, farklı uzmanlık alanlarına sahip iki ekip olarak düşünebiliriz:

1.  **Ekip 1 - Saha Ajanları (1. Gizli Katman):** Bu 64 nöronun görevi, ham veriye (ajanın x,y konumu) bakıp basit ve temel desenleri tanımaktır. Her biri farklı bir konuda uzmanlaşmıştır:
    *   *Nöron A:* "Ajan sağ duvara ne kadar yakın?" diye sorar.
    *   *Nöron B:* "Ajan üst sıralarda mı?" diye kontrol eder.
    *   *Nöron C:* "Ajan tam ortada mı?" diye analiz eder.
    
    Hepsi aynı `(x, y)` bilgisine bakar ama her biri kendi uzmanlığına göre farklı bir rapor çıkarır.

2.  **Ekip 2 - Analistler (2. Gizli Katman):** Bu 64 nöronun görevi daha karmaşıktır. Saha ajanlarından gelen 64 farklı raporu masaya yatırıp bunları birleştirerek daha üst düzey, daha soyut sonuçlar çıkarmaktır.
    *   *Analist X:* Nöron A'nın "sağ duvara yakın" raporu ile Nöron B'nin "üst sıralarda" raporunu birleştirip "Ajan sağ üst köşeye yakın bir bölgede, buradan hedefe gitmek için aşağı ve sola gitmek mantıklı" gibi bir sonuca varabilir.

### Nöron Nasıl Çalışır?

Her bir nöronun içinde iki adımlı bir süreç işler:

1.  **Görüş Oluşturma (Ağırlıklı Toplam):** Her bir nöronun, eğitim sırasında öğrendiği kişisel bir "ilgi listesi" (ağırlıklar - `W`) ve bir "ön yargısı" (bias - `b`) vardır. Bir önceki ekipten gelen raporları kendi ilgi listesine göre tartar, toplar ve sonuca kendi ön yargısını ekler. Matematiksel olarak `(girdi @ ağırlıklar) + bias` işlemi budur.

2.  **Karar Verme (Aktivasyon Fonksiyonu - ReLU):** Nöron, oluşturduğu bu ham görüşe bakar ve bir karar verir: "Bu bilgi bir sonraki ekibe aktarılacak kadar önemli mi?" Bu kararı **ReLU** adında çok basit bir kuralla verir:
    *   **"Eğer hesapladığım sayı pozitifse, önemlidir, aynen söyle. Eğer negatif veya sıfırsa, önemsizdir, '0' de ve konuyu kapat."**

Her bir nöronda bambaşka değerler görmemizin sebebi işte bu: **Her bir nöronun kendine ait, eğitimde öğrendiği tamamen FARKLI bir ilgi listesi (ağırlıklar) ve ön yargısı (bias) vardır.** Bu çeşitlilik sayesinde ağ, bir problemi yüzlerce farklı açıdan analiz edebilir.

### Ağırlık Başlatma

Bir sinir ağını eğitirken, nöronların işe nereden başlayacağı çok önemlidir. `MLP` sınıfında şu satırı görürüz:
```
... * np.sqrt(2.0/in_dim)
```

Bu, **"He Başlatma Yöntemi"** adı verilen bir tekniktir. Sezgisel olarak anlamı şudur: Nöronlara işe başlarken ne çok iyimser (çok büyük sayılar) ne de çok kötümser (çok küçük sayılar) olmamalarını söylemektir. Onlara "sağlam ve dengeli bir başlangıç noktası" veririz. Bu, eğitimin en başında ağın kilitlenmesini veya kararsızlaşmasını engelleyen kritik bir adımdır.

---

## Öğrenme Mekanizmaları

DQN'in öğrenme sürecini optimize eden iki temel mekanizma:

### Experience Replay: Ajanın Hafızası

Ajan, yaşadığı her deneyimi `(durum, eylem, ödül, yeni_durum)` hemen kullanıp unutmaz. Bunun yerine, `ReplayBuffer` adını verdiğimiz büyük bir hafızada biriktirir. Neden?

1.  **"Deja Vu" Yaşatmak:** Ajan art arda 5 adım sağa gittiyse, ağı sadece "sağa gitmek" üzerine düşünmeye başlar. Bu dar bakış açısını kırmak için, hafızasından rastgele anılar çekeriz: biraz geçmişten, biraz farklı bir yerden... Bu, ağın daha genel ve esnek bir strateji öğrenmesini sağlar.

2.  **Deneyimden Tasarruf:** Yaşanan her bir anı çok değerlidir. Bu anıyı hafızaya atarak, ağın aynı dersi tekrar tekrar öğrenmesi için defalarca kullanabiliriz. Bu, öğrenmeyi inanılmaz verimli hale getirir.

### Target Network: Sabırlı Öğretmen

Eğitim sırasında ağın (policy network) kendisi sürekli değişir. Bu, öğrenmeye çalıştığı hedefin de sürekli değişmesi demektir. "Hareketli bir hedefi vurmaya çalışmak" gibi düşün. Bu, öğrenmeyi zorlaştırır.

Çözümümüz: İki ağ kullanırız.
- **Öğrenci Ağ (`policy`):** Her adımda öğrenmeye çalışan, sürekli güncellenen ana ağ.
- **Öğretmen Ağ (`target`):** Öğrencinin belirli aralıklarla (örn: 200 adımda bir) kopyasını aldığı, yavaş ve sabırlı olan ikinci ağ.

Öğrenci, hedefinin ne olduğunu belirlerken bu sakin ve sabırlı öğretmen ağa bakar. Bu, ona daha istikrarlı bir hedef sunar ve kafasının karışmasını engeller.

---

## Özet

`deep_q_network_numpy.py` dosyasındaki kod, yaşayan, düşünen bir mekanizmanın somut bir temsilidir:

- **Gözler (Girdi):** Dünyayı algılar.
- **Uzman Ekipleri (Gizli Katmanlar):** Algıyı anlamlı ipuçlarına dönüştürür.
- **Karar Merkezi (Çıktı):** İpuçlarını birleştirip bir eylem seçer.
- **Hafıza (Replay Buffer):** Geçmişten ders çıkarır.
- **Öğretmen (Target Network):** Öğrenme yolculuğunu dengede tutar.

