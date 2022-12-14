# DQN Nasıl Çalışır?

---

## MLP: Ajanın Beyni

Her şeyden önce, ajanımızın bir beyni var. Bu beyin, `MLP` (Multi-Layer Perceptron) adını verdiğimiz bir sinir ağı. Bu beynin görevi çok net: Ajanın nerede olduğunu öğrenip, her bir olası hamle için bir "iyilik puanı" (Q-değeri) vermek.

Peki bu beyin nasıl çalışıyor? Onu bir soru-cevap makinesi gibi düşünelim.

### Girdi ve Çıktı Katmanları

- **Soru (Girdi - 2 Nöron):** Beyne sorduğumuz soru çok basit: "Sevgili ajan, sen şu anda `(x, y)` konumundasın. Ne yapmalıyız?"
  - Ajanın 2 boyutlu haritadaki konumunu (`x` ve `y`) temsil etmek için **2 nörona** ihtiyacımız var. Bunlar, beynin dünyaya açılan "gözleri"dir.

- **Cevap (Çıktı - 4 Nöron):** Beynin verdiği cevap da bir o kadar net: "Bulunduğun konumda, potansiyel hamlelerinin puanları şunlar:"
  - Ajanın yapabileceği 4 temel eylem (Yukarı, Aşağı, Sol, Sağ) olduğu için, her bir eylemin puanını aynı anda söyleyebilmesi için **4 nörona** ihtiyacı var.
  - Bize `[yukarı_puanı, aşağı_puanı, sol_puanı, sağ_puanı]` gibi bir liste verir. Biz de en yüksek puanlıyı seçeriz.

### Gizli Katmanlar

Girdi (soru) ve çıktı (cevap) arasındaki o 64'er nöronluk iki gizli katman var ya... İşte beynin asıl "düşünme" ve "akıl yürütme" eylemini gerçekleştirdiği yer orası. Onları iki farklı uzmanlık alanına sahip iki ekip olarak hayal edelim:

1.  **Ekip 1 - Saha Ajanları (1. Gizli Katman):** Bu 64 "saha ajanı" nöronun görevi, ham veriye (ajanın x,y konumu) bakıp basit ve temel desenleri tanımaktır. Her biri farklı bir konuda uzmanlaşmıştır:
    *   *Nöron A:* "Ajan sağ duvara ne kadar yakın?" diye sorar.
    *   *Nöron B:* "Ajan üst sıralarda mı?" diye kontrol eder.
    *   *Nöron C:* "Ajan tam ortada mı?" diye analiz eder.
    Hepsi aynı `(x, y)` bilgisine bakar ama her biri kendi uzmanlığına göre farklı bir rapor çıkarır.

2.  **Ekip 2 - Analistler (2. Gizli Katman):** Bu 64 "analist" nöronun görevi daha karmaşıktır. Saha ajanlarından gelen 64 farklı raporu masaya yatırıp bunları birleştirerek daha üst düzey, daha soyut sonuçlar çıkarmaktır.
    *   *Analist X:* Nöron A'nın "sağ duvara yakın" raporu ile Nöron B'nin "üst sıralarda" raporunu birleştirip "Demek ki ajan sağ üst köşeye yakın bir tehlike bölgesinde!" sonucuna varabilir.

### Nöron Nasıl Çalışır?

Peki, o uzmanlardan sadece bir tanesi bir değeri nasıl hesaplıyor? Her bir nöronun içinde iki adımlı basit bir süreç işler:

1.  **Görüş Oluşturma (Ağırlıklı Toplam):** Her bir uzmanın, eğitim sırasında öğrendiği kişisel bir "ilgi listesi" (ağırlıklar - `W`) ve bir "ön yargısı" (bias - `b`) vardır. Bir önceki ekipten gelen raporları kendi ilgi listesine göre tartar, toplar ve sonuca kendi ön yargısını ekler. Bu, uzmanın ilk "ham görüşüdür". Matematiksel olarak `(girdi @ ağırlıklar) + bias` işlemi budur.

2.  **Karar Verme (Aktivasyon Fonksiyonu - ReLU):** Uzman, oluşturduğu bu ham görüşe bakar ve bir karar verir: "Bu bilgi bir sonraki ekibe aktarılacak kadar önemli mi?" Bu kararı **ReLU** adında çok basit bir kuralla verir:
    *   **"Eğer hesapladığım sayı pozitifse, önemlidir, aynen söyle. Eğer negatif veya sıfırsa, önemsizdir, '0' de ve konuyu kapat."**

Her bir nöronda bambaşka değerler görmemizin sebebi işte bu: **Her bir uzmanın kendine ait, eğitimde öğrendiği tamamen FARKLI bir ilgi listesi (ağırlıklar) ve ön yargısı (bias) vardır.** Bu çeşitlilik sayesinde beyin, bir problemi yüzlerce farklı açıdan analiz edebilir.

### Ağırlık Başlatma

Bir sinir ağını eğitirken, bu uzmanların işe nereden başlayacağı çok önemlidir. `MLP` sınıfının başında şu satırı görürüz:
`... * np.sqrt(2.0/in_dim)`

Bu, **"He Başlatma Yöntemi"** adı verilen bir tekniktir. Sezgisel olarak anlamı şudur: Uzmanlara işe başlarken ne çok iyimser (çok büyük sayılar) ne de çok kötümser (çok küçük sayılar) olmamalarını söylemektir. Onlara "sağlam ve dengeli bir başlangıç noktası" veririz. Bu, eğitimin en başında beynin kilitlenmesini veya kararsızlaşmasını engelleyen kritik bir adımdır.

---

## Öğrenme Mekanizmaları

Bu beyni daha da akıllı yapan iki temel mekanizma var:

### Replay Buffer: Ajanın Hafızası

Ajan, yaşadığı her deneyimi `(durum, eylem, ödül, yeni_durum)` hemen kullanıp unutmaz. Bunun yerine, `ReplayBuffer` adını verdiğimiz büyük bir "hafıza"da biriktirir. Neden?

1.  **"Deja Vu" Yaşatmak:** Ajan art arda 5 adım sağa gittiyse, beyni sadece "sağa gitmek" üzerine düşünmeye başlar. Bu dar bakış açısını kırmak için, hafızasından rastgele anılar çekeriz: biraz geçmişten, biraz farklı bir yerden... Bu, beynin daha genel ve esnek bir strateji öğrenmesini sağlar.
2.  **Deneyimden Tasarruf:** Yaşanan her bir anı çok değerlidir. Bu anıyı hafızaya atarak, beynin aynı dersi tekrar tekrar öğrenmesi için defalarca kullanabiliriz. Bu, öğrenmeyi inanılmaz verimli hale getirir.

### Target Network: Sabırlı Öğretmen

Eğitim sırasında beynin (ana ağ) kendisi sürekli değişir. Bu, öğrenmeye çalıştığı hedefin de sürekli değişmesi demektir. "Hareketli bir hedefi vurmaya çalışmak" gibi düşün. Bu, öğrenmeyi zorlaştırır.

Çözümümüz dahice: İki beyin kullanırız.
- **Öğrenci Beyin (`policy`):** Her adımda deli gibi öğrenmeye çalışan, sürekli güncellenen ana beyin.
- **Öğretmen Beyin (`target`):** Öğrencinin belirli aralıklarla (örn: 200 adımda bir) kopyasını aldığı, yavaş ve sabırlı olan ikinci beyin.

Öğrenci, hedefinin ne olduğunu belirlerken bu sakin ve sabırlı "öğretmen" beyne bakar. Bu, ona daha istikrarlı bir hedef sunar ve kafasının karışmasını engeller.

---

## Özet

Artık `deep_q_network_numpy.py` dosyasındaki kod sadece bir dizi komut değil; yaşayan, düşünen bir mekanizmanın somut bir temsilidir:

- **Gözler (Girdi):** Dünyayı algılar.
- **Uzman Ekipleri (Gizli Katmanlar):** Algıyı anlamlı ipuçlarına dönüştürür.
- **Karar Merkezi (Çıktı):** İpuçlarını birleştirip bir eylem seçer.
- **Hafıza (Replay Buffer):** Geçmişten ders çıkarır.
- **Öğretmen (Target Network):** Öğrenme yolculuğunu dengede tutar.

Umarım bu yolculuk, bir DQN ajanının zihninde neler olup bittiğini daha sezgisel bir şekilde anlamana yardımcı olmuştur!
