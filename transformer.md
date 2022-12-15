# Transformer Nasıl Çalışır?

Transformer mimarisi, self-attention mekanizması ve positional encoding kullanarak özellikler arasındaki karmaşık ilişkileri öğrenir. Bu yaklaşım, LSTM'in sıralı işleme sınırlamalarını aşarak paralel hesaplama ve uzun menzilli bağımlılıkları modelleme yeteneği sağlar.

---

## Özellik Sequence Temsili

Transformer, durumu bir özellik sequence'i olarak temsil eder. Her özellik bir "token" olarak düşünülür:

- **Token 0:** Ajan X koordinatı
- **Token 1:** Ajan Y koordinatı  
- **Token 2:** Hedef X koordinatı
- **Token 3:** Hedef Y koordinatı
- **Token 4:** Ajan-hedef mesafesi

Bu token'lar, self-attention mekanizması ile birbirleriyle etkileşime girer ve aralarındaki ilişkileri öğrenir.

---

## Self-Attention Mekanizması

Self-attention, her token'ın diğer tüm token'larla nasıl ilişkili olduğunu öğrenir. Bu, Query (Q), Key (K) ve Value (V) matrisleri üzerinden gerçekleşir:

### Query, Key, Value Üçlüsü

- **Query (Q):** "Ne arıyorum?" - Her token'ın sorgusu
- **Key (K):** "Ne sunuyorum?" - Her token'ın sağladığı bilgi
- **Value (V):** "Gerçek içerik" - Her token'ın taşıdığı değer

Her token, Q matrisi ile diğer token'ların K matrislerini karşılaştırarak hangi token'lara dikkat etmesi gerektiğini öğrenir. Benzerlik skorları hesaplanır ve bu skorlar softmax ile normalize edilerek attention weight'leri elde edilir.

### Scaled Dot-Product Attention

Attention hesaplaması şu formülle yapılır:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Burada `sqrt(d_k)` ile ölçekleme yapılması, gradient'lerin büyük değerler almasını önler ve eğitimi stabilize eder.

### Özellikler Arası İlişkiler

Self-attention sayesinde Transformer şu tür ilişkileri öğrenebilir:
- Ajan X ve Hedef X koordinatları arasındaki yatay hizalama
- Ajan Y ve Hedef Y koordinatları arasındaki dikey hizalama  
- Mesafe bilgisinin diğer özelliklerle kombinasyonu
- Koordinatlar ve mesafe arasındaki geometrik ilişkiler

---

## Positional Encoding

Self-attention mekanizması token'ların sırasını bilmez. Bu eksikliği gidermek için positional encoding kullanılır. Her token'ın pozisyonunu kodlayan sinüs ve kosinüs dalgaları eklenir:

### Sinusoidal Encoding

Positional encoding, sinüs ve kosinüs fonksiyonları kullanılarak her pozisyon için benzersiz bir kodlama oluşturur:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Bu yaklaşım, token'ların hem mutlak hem de göreceli pozisyonlarını kodlar. Pozisyonlar yakınsa encoding'ler benzer, uzaksa farklı olur.

### Embedding ile Birleştirme

Token embedding'leri ile positional encoding toplama ile birleştirilir. Bu sayede her token hem içeriğini (embedding) hem de pozisyonunu (encoding) taşır.

---

## Multi-Head Attention

Multi-head attention, aynı bilgiyi birden fazla farklı "head" (kafa) ile paralel olarak analiz eder. Her head kendi Q, K, V matrislerine sahiptir:

### Paralel Analiz

- **Head 1:** Koordinat ilişkilerine odaklanabilir (Ajan X ↔ Hedef X)
- **Head 2:** Mesafe ve pozisyon ilişkilerine odaklanabilir
- **Head 3:** Genel geometrik desenleri yakalayabilir
- **Head 4:** Farklı bir açıdan özellik kombinasyonlarını analiz edebilir

Her head'in çıktıları concatenate edilir ve bir lineer katman üzerinden geçirilir. Bu sayede Transformer, bilgiyi çok yönlü analiz edebilir.

---

## Transformer Layer Yapısı

Her Transformer layer iki ana bileşenden oluşur:

### 1. Multi-Head Self-Attention

Özellikler arası ilişkileri öğrenen ana mekanizma. Residual connection ve layer normalization ile birlikte kullanılır.

### 2. Feed-Forward Network

Attention çıktısını işleyen iki katmanlı bir MLP. Bu katman, attention'ın öğrendiği ilişkileri daha da geliştirir.

### Residual Connection ve Layer Normalization

Her alt-katman çevresinde residual connection ve layer normalization kullanılır. Bu:
- Gradient akışını iyileştirir
- Eğitimi stabilize eder
- Daha derin ağların eğitilmesini mümkün kılar

---

## LSTM ve Attention'dan Farkları

### LSTM'den Farkları

- **Paralel Hesaplama:** Transformer tüm token'ları paralel işler, LSTM sıralı işler
- **Uzun Menzil Bağımlılıkları:** Her token, uzaklıktan bağımsız olarak tüm diğer token'lara direkt erişebilir
- **Öğrenilen Pozisyon:** Positional encoding öğrenilmez, hesaplanır (alternatif olarak öğrenilebilir de olabilir)

### Attention Mekanizmasından Farkları

- **Kapsamlı Mimari:** Transformer, sadece attention değil, aynı zamanda layer yapısı, feed-forward network ve positional encoding içerir
- **Katmanlı Yapı:** Birden fazla Transformer layer'ı ile hiyerarşik özellik öğrenimi sağlar
- **Daha Güçlü Temsil:** Multi-layer yapı sayesinde daha kompleks özellik kombinasyonları öğrenir

---

## GridWorld'deki Kullanımı

Transformer, GridWorld problemini çözerken:

- **Koordinat İlişkilerini Öğrenir:** Ajan ve hedef koordinatları arasındaki geometrik ilişkileri self-attention ile modelle
- **Mesafe Bilgisini Entegre Eder:** Mesafe token'ı ile diğer özellikler arasındaki bağlantıları öğrenir
- **Karmaşık Stratejiler:** Multi-layer yapı sayesinde basit özelliklerden karmaşık stratejilere kadar öğrenebilir
- **Pozisyon Farkındalığı:** Positional encoding sayesinde hangi özelliğin nerede olduğunu bilir


