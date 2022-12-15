# LSTM Nasıl Çalışır?

---

## Hafıza Mekanizması

MLP ve CNN, sadece **mevcut duruma** bakar. "Ajan şu anda (2,1) konumunda, ne yapmalı?" diye sorar ve cevap verir. Ama LSTM çok daha zengin bir bilgiye sahiptir: **Geçmişi hatırlar!**

LSTM, "5 adım önce neredeydim?" sorusunu cevaplayabilir. Bu sayede:
- Döngülerden kaçınır (aynı yere tekrar tekrar gitmez)
- Uzun vadeli stratejiler geliştirir
- Geçmiş hareketlerini analiz ederek daha iyi kararlar verir

### Sequence: Zaman İçinde Durumlar

LSTM, durumları **sequence** (dizi) olarak alır. Örneğin, son 10 adımın durumlarını bir araya getirir:

```
Sequence = [
    durum_t-9,  # 9 adım önce
    durum_t-8,  # 8 adım önce
    ...
    durum_t-1,  # 1 adım önce
    durum_t     # Şu an
]
```

Bu sequence, LSTM'in "belleğidir". Tıpkı bir insanın "son 10 saniyede ne yaptım?" diye düşünmesi gibi, LSTM de "son 10 adımda neredeydim?" diye sorar.

### LSTM Hücresi

LSTM'in kalbi, **LSTM hücresi**dir. Bu hücre, bilgiyi nasıl hatırlayacağını, nasıl unutacağını ve nasıl kullanacağını öğrenir. Bunu yapmak için **4 gate (kapı)** kullanır:

1. **Forget Gate (Unutma Kapısı):** "Eski bilgiyi ne kadar unutmalıyım?"
2. **Input Gate (Girdi Kapısı):** "Yeni bilgiyi ne kadar hafızaya almalıyım?"
3. **Candidate Gate (Aday Kapısı):** "Yeni bilginin değeri nedir?"
4. **Output Gate (Çıktı Kapısı):** "Hangi bilgiyi dışarı çıkarayım?"

Her gate, 0 ile 1 arasında bir değer üretir. 0 = "kapalı" (bilgi geçmez), 1 = "tam açık" (tüm bilgi geçer).

### Cell State: Uzun Vadeli Hafıza

Cell state, LSTM'in **uzun vadeli hafızasıdır**. Bu, tıpkı bir kişinin "genel birikimi" gibidir. Cell state, zaman içinde yavaşça değişir ve önemli bilgileri uzun süre saklar.

Örnek: "Ajan, sol üst köşeye yakın bir yerde döngüye girdi" bilgisi, cell state'te saklanır. Bu sayede ajan, bir sonraki sefer aynı duruma geldiğinde farklı bir yol seçer.

### Hidden State: Kısa Vadeli Hafıza

Hidden state, LSTM'in **kısa vadeli hafızasıdır**. Bu, mevcut durumla ilgili bilgileri taşır ve her adımda güncellenir.

Cell state ve hidden state birlikte çalışır:
- Cell state: "Genel strateji" (uzun vadeli)
- Hidden state: "Mevcut durum" (kısa vadeli)

---

## Gate'ler Nasıl Çalışır?

### Forget Gate: Ne Zaman Unutmalı?

Forget gate, **eski bilgiyi ne kadar unutmalıyım?** sorusunu cevaplar.

Örnek senaryo: Ajan 5 adım önce (0,0) konumundaydı. Şimdi (3,2) konumunda. "5 adım önceki konum" bilgisi artık önemli mi? Belki değil. Forget gate, bu bilgiyi **unutur** (0'a yakın değer üretir).

Ama eğer ajan, "sol üst köşede döngüye girdi" gibi önemli bir bilgi öğrendiyse, forget gate bunu **korur** (1'e yakın değer üretir).

**Matematiksel olarak:**
```
f_t = sigmoid([mevcut_durum, önceki_hidden_state] @ W_f + b_f)
c_t = f_t * c_prev + ...  # Eski cell state'i forget gate ile filtrele
```

### Input Gate: Ne Zaman Öğrenmeli?

Input gate, **yeni bilgiyi ne kadar hafızaya almalıyım?** sorusunu cevaplar.

Örnek senaryo: Ajan yeni bir konuma geldi ve "bu konumdan hedefe giden kısa bir yol var" bilgisini öğrendi. Bu bilgi önemli mi? Evet! Input gate, bu bilgiyi hafızaya **kaydeder** (1'e yakın değer üretir).

Ama eğer ajan sadece rastgele bir hareket yaptıysa, input gate bu bilgiyi **görmezden gelir** (0'a yakın değer üretir).

**Matematiksel olarak:**
```
i_t = sigmoid([mevcut_durum, önceki_hidden_state] @ W_i + b_i)
g_t = tanh([mevcut_durum, önceki_hidden_state] @ W_g + b_g)  # Yeni bilgi
c_t = ... + i_t * g_t  # Yeni bilgiyi input gate ile filtreleyerek ekle
```

### Candidate Gate: Yeni Bilginin Değeri

Candidate gate, **yeni bilginin değeri nedir?** sorusunu cevaplar. Bu, input gate ile çarpılarak cell state'e eklenir.

Tanh aktivasyonu kullanılır çünkü hem pozitif hem negatif değerlere izin verir:
- Pozitif değer: "Bu bilgi faydalı"
- Negatif değer: "Bu bilgi zararlı veya önemsiz"

### Output Gate: Hangi Bilgiyi Dışarı Çıkarayım?

Output gate, **cell state'ten ne kadarını dışarı çıkarayım?** sorusunu cevaplar.

Cell state, çok fazla bilgi içerir. Ama her zaman tüm bilgiye ihtiyacımız yok. Output gate, sadece **gerekli bilgiyi** hidden state olarak dışarı çıkarır.

**Matematiksel olarak:**
```
o_t = sigmoid([mevcut_durum, önceki_hidden_state] @ W_o + b_o)
h_t = o_t * tanh(c_t)  # Cell state'i output gate ile filtrele
```

---

## MLP ve CNN'den Farkı

### Zaman Bağımlılığı

- **MLP/CNN:** Sadece mevcut duruma bakar. "Şu anda (2,1)'deyim, ne yapmalıyım?"
- **LSTM:** Geçmiş durumları da hatırlar. "Şu anda (2,1)'deyim, ama 5 adım önce de buradaydım. Belki bir döngüye girdim, farklı bir yol denemeliyim!"

### Döngü Önleme

MLP ve CNN, aynı duruma geldiğinde **aynı eylemi** seçer. Bu, döngülere yol açabilir:
- (0,0) → Sağa → (1,0) → Aşağı → (1,1) → Sola → (0,1) → Yukarı → (0,0) → **Döngü!**

LSTM ise, "5 adım önce de (0,0)'daydım, tekrar buraya dönmemeliyim" diye düşünür ve farklı bir eylem seçer.

### Uzun Vadeli Strateji

LSTM, uzun vadeli stratejiler geliştirebilir:
- "Önce sağa git, sonra aşağı git, sonra tekrar sağa git" gibi karmaşık yollar öğrenebilir.
- Geçmiş hareketlerini analiz ederek, "en kısa yol nedir?" sorusunu cevaplayabilir.

---

## Öğrenme Süreci

### İlk Adımlar: Keşif

Başlangıçta gate'ler rastgele ağırlıklara sahiptir. Henüz neyi hatırlaması, neyi unutması gerektiğini bilmiyorlar. Ajan rastgele hareket eder ve sequence'ler oluşturur.

### Gate'lerin Uzmanlaşması

Eğitim ilerledikçe, gate'ler kendi uzmanlıklarını geliştirir:

- **Forget Gate:** "Eski konumlar önemli değil, ama öğrenilen stratejiler önemli" diye öğrenir.
- **Input Gate:** "Yeni bilgi kaynağı (hedefe yakınlık, döngü sinyalleri) önemli" diye öğrenir.
- **Output Gate:** "Cell state'ten sadece eylem seçimi için gerekli bilgiyi çıkar" diye öğrenir.

### Cell State'in Anlamı

Eğitim sonrası cell state'e baktığımızda:
- Bazı değerler yüksek kalır (uzun vadeli stratejiler)
- Bazı değerler sürekli güncellenir (kısa vadeli bilgiler)
- Bazı değerler sıfırlanır (unutulan bilgiler)

Bu, LSTM'in "düşündüğü şeyi görmemize" izin verir. Sanki LSTM'in zihnini okuyormuş gibi!

---

## Özet

LSTM, GridWorld problemini çözerken:

- **Geçmişi hatırlar:** Son N adımın durumlarını sequence olarak saklar.
- **Akıllı unutma:** Forget gate ile önemli bilgiyi korur, önemsiz bilgiyi siler.
- **Yeni bilgi öğrenme:** Input gate ile önemli bilgiyi hafızaya kaydeder.
- **Döngü önleme:** Geçmiş hareketleri analiz ederek aynı yere tekrar gitmekten kaçınır.
- **Uzun vadeli strateji:** Cell state sayesinde karmaşık yollar öğrenir.

LSTM, MLP ve CNN'den farklı olarak, sadece mevcut durumu değil, geçmiş durumları da kullanarak karar verir. Bu sayede zaman bağımlı desenleri öğrenir ve uzun vadeli stratejiler geliştirir.


