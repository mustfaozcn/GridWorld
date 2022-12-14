# Q-Learning Nasıl Çalışır?

Bu dosya, `q_learning.py` kodunun nasıl çalıştığını açıklıyor. DQN'den farklı olarak Q-Learning daha basit bir yaklaşım: sinir ağı yerine bir tablo kullanır.

DQN'deki ajanımız "akıl yürüten" bir beyne (sinir ağı) sahipti. Buradaki ajanımız ise daha çok çalışkan bir öğrenci gibi; her şeyi bir **"kopya kağıdına" (Q-Tablosu)** yazarak ezberliyor. Gelin bu ajanın kopya kağıdını nasıl doldurduğunu inceleyelim.

---

## Q-Tablosu Nedir?

Bu ajanın beyni, `MLP` gibi karmaşık bir yapı değil. Sadece dev bir NumPy dizisi, yani bir tablo.

```python
Q = np.full((env.h, env.w, 4), ...)
```

Bu tablonun her bir hücresi, ajan için çok basit bir bilgiyi saklar:
**"Eğer `(x, y)` konumundaysan ve `[eylem]` yaparsan, almayı beklediğin toplam puan budur."**

-   **Katman veya Nöron Yok:** Ajan bir şey "hesaplamaz" veya "tahmin etmez".
-   **Doğrudan Bakar:** Karar vermek için, mevcut konumuna (`x, y`) karşılık gelen satırı tablodan bulur, 4 eylemin puanlarına bakar ve en yüksek puanlıyı seçer. Tıpkı bir çarpım tablosuna bakmak gibi.

Bu yüzden bu yönteme **Tabular (Tablosal) Q-Learning** denir.

---

## Q-Tablosu Nasıl Öğrenir?

Ajan, bu boş hesap tablosunu en doğru bilgilerle nasıl doldurur? İşte öğrenmenin kalbi olan o meşhur formül burada devreye giriyor:

```python
Q[y, x, a] += alpha * td_error
```

Bu formülü parçalara ayıralım:

-   `Q[y, x, a]`: Bu, kopya kağıdındaki değer. Yani, `(x, y)` konumunda `a` eylemini yapmanın "eski" puanı.
-   `+=`: "Eski puanı, birazdan hesaplayacağımız yeni bilgiyle güncelle."
-   `alpha`: Bu bizim **öğrenme oranımız**. Sezgisel olarak, "yeni bilgiye ne kadar inanmalıyım?" sorusunun cevabıdır.
    -   Eğer `alpha` küçükse, ajan tutucudur: "Bu yeni bilgi ilginç ama ben bildiğime şimdilik daha çok güveniyorum."
    -   Eğer `alpha` büyükse, ajan heveslidir: "Vay canına, bu yeni bilgi harika! Eski bilgiyi büyük ölçüde bununla değiştireyim."
-   `td_error`: Bu, **"sürpriz faktörüdür"** (Temporal Difference Error). Ajanın beklentisi ile gerçeğin ne kadar farklı olduğudur.

### Sürpriz Faktörü (`td_error`) Nasıl Hesaplanır?

`td_error = td_target - Q[y, x, a]`

-   `td_target`: Bu, ajanın o adımda ulaşmayı **umduğu ideal puandır**. Şöyle hesaplanır:
    `r + gamma * np.max(Q[ny, nx, :])`
    -   `r`: O adımı atınca **hemen kazandığı ödül** (ya -1 ya da +10).
    -   `gamma`: Ajanın **sabırsızlık katsayısıdır**. Gelecekteki ödüllere ne kadar önem verdiğini belirler. 1'e ne kadar yakınsa, o kadar sabırlı ve ileri görüşlüdür.
    -   `np.max(Q[ny, nx, :])`: Bu kısım, "attığım adımdan sonra geldiğim **yeni konumda**, kopya kağıdıma göre yapabileceğim **en iyi hamlenin puanı nedir?**" sorusunun cevabıdır.

**Özetle Öğrenme Süreci:**

1.  **Beklenti:** Ajan, kopya kağıdına bakar ve `(x, y)`'de `a` eylemini yapmanın puanının `Q[y, x, a]` olduğunu düşünür.
2.  **Eylem ve Sonuç:** Eylemi yapar, `r` ödülünü alır ve yeni bir `(nx, ny)` konumuna gelir.
3.  **Değerlendirme (Sürpriz):** Kendi kendine sorar: "Kazandığım anlık ödül (`r`) ile vardığım yeni yerdeki en iyi hamlenin potansiyelini (`gamma * max(Q)`) toplarsam, bu eylemin gerçek değeri ne olmalıydı?" Bu, onun `td_target`'ı olur.
4.  **Güncelleme:** "Gerçek değer (`td_target`) ile benim eski tahminim (`Q[y, x, a]`) arasında ne kadar fark var?" Bu fark, onun "sürprizi" (`td_error`) olur. Kopya kağıdındaki eski değeri, bu sürprizin `alpha` kadarını ekleyerek günceller.

Bu süreç binlerce kez tekrarlandığında, tablodaki değerler yavaş yavaş gerçeğe yakınsar ve ajan en iyi yolu ezberlemiş olur.

---

## Keşif ve Kullanım Dengesi

Eğer ajan her zaman sadece kopya kağıdındaki en yüksek puanlı eylemi seçerse, belki de daha iyi olan ama daha önce hiç denemediği bir yolu asla bulamaz.

İşte burada **epsilon (`ε`)** devreye girer. Epsilon, ajanın **"delilik" veya "merak" seviyesidir**.

-   **`epsilon` olasılıkla (Örn: %10):** Ajan kopya kağıdını bir kenara atar ve "Bugün de bir delilik yapayım!" diyerek tamamen **rastgele** bir eylem seçer. Bu, yeni yollar **keşfetmesini** sağlar.
-   **`1-epsilon` olasılıkla (Örn: %90):** Ajan aklı başında davranır ve kopya kağıdındaki en yüksek puanlı eylemi seçerek bildiği en iyi yolu **kullanır (exploitation)**.

Eğitimin başında `epsilon` yüksektir (ajan çok meraklıdır), zamanla azalır (ajan öğrendiklerine daha çok güvenmeye başlar).

Bu basit ama güçlü mekanizmalarla, bir Q-Tablosu ajanı, karmaşık bir sinir ağına ihtiyaç duymadan, dünyayı deneyimleyerek ve sonuçları bir tabloya dikkatlice not ederek en iyi yolu bulmayı öğrenir.
