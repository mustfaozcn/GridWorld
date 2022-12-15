# ResNet Nasıl Çalışır?

ResNet (Residual Network), derin sinir ağlarında ortaya çıkan vanishing gradient problemini skip connection (atlamalı bağlantı) mekanizması ile çözer. Bu yaklaşım, gradient'lerin katmanlar arasında daha etkili bir şekilde akmasına izin vererek çok daha derin ağların eğitilmesini mümkün kılar.

---

## Vanishing Gradient Problemi

Geleneksel derin ağlarda, gradient'ler geri yayılım sırasında katmanlardan geçerken küçülme eğilimi gösterir. Özellikle sigmoid veya tanh aktivasyon fonksiyonları kullanıldığında, her katman gradient'i büyük ölçüde azaltır. Sonuç olarak:

- **Erken katmanlar:** Gradient'ler çok küçük olur, ağırlık güncellemeleri minimal kalır
- **Öğrenme yavaşlar:** Alt katmanlar öğrenemez, sadece üst katmanlar güncellenir
- **Derin ağlar eğitilemez:** 3-4 katmandan fazla derin ağlar pratikte eğitilemez hale gelir

ReLU aktivasyon fonksiyonu bu sorunu kısmen çözer ama yeterli değildir. ResNet, bu probleme köklü bir çözüm sunar.

---

## Residual Block ve Skip Connection

ResNet'in temel yapı taşı **residual block**'tur. Her residual block, şu formülü uygular:

```
output = F(x) + x
```

Burada:
- **F(x):** İki katmanlı bir transformasyon (öğrenilen değişiklik/residual)
  - İlk katman: `z1 = x @ W1 + b1`, ardından ReLU
  - İkinci katman: `z2 = a1 @ W2 + b2`, ardından ReLU
  - `F(x) = ReLU(z2)`
- **x:** Orijinal girdi (skip connection ile direkt geçer)
- **output:** Transformasyon + orijinal girdi (residual + identity)

### Identity Mapping

Eğer ağ, optimal çözümün girdiyle aynı olması gerektiğini öğrenirse, `F(x) = 0` öğrenmesi yeterlidir. Bu, **identity mapping** olarak bilinir ve derin ağların en azından girdi kadar iyi performans göstermesini garanti eder.

Skip connection sayesinde:
- Ağ, önemsiz transformasyonları öğrenebilir (`F(x) ≈ 0`)
- Daha karmaşık transformasyonları da öğrenebilir (`F(x) ≠ 0`)
- Her iki durumda da gradient akışı kesintisiz kalır

---

## Gradyan Akışı ve Skip Connection

Skip connection'ın en kritik faydası, gradient akışını korumasıdır:

### Normal Ağda Gradient Akışı

Geri yayılım sırasında, gradient her katmandan geçerken küçülür:
```
dL/dW1 = (dL/dy) * (dy/dW1)  // Gradient küçülür
dL/dW2 = (dL/dW1) * (dW1/dW2)  // Daha da küçülür
```

Çok katmanlı yapılarda bu gradient sıfıra yakınlaşır ve alt katmanlar öğrenemez.

### ResNet'te Gradient Akışı

Skip connection sayesinde, gradient iki yoldan geçer:
1. **Transformasyon yolu:** `F(x)` üzerinden (küçülebilir)
2. **Skip connection yolu:** Direkt `x` üzerinden (korunur)

Gradient formülü:
```
dL/dx = dL/doutput * (dF/dx + 1)
```

`+1` terimi, skip connection'dan gelen gradient'in korunduğunu garantiler. Bu sayede alt katmanlar bile güncellenebilir.

---

## ResNet Mimarisi

ResNet mimarisi üç ana bileşenden oluşur:

### 1. Giriş Projeksiyonu

Girdiyi (ajanın konumu) gizli boyuta projekte eder:
```
x → W_in @ x + b_in → hidden_dim boyutlu vektör
```

### 2. Residual Block'lar

Birden fazla residual block art arda sıralanır:
```
x → Block 1 → Block 2 → Block 3 → ... → Block N
```

Her block:
- Girdiyi transform eder: `F(x)`
- Girdiyi direkt ekler: `x + F(x)`
- ReLU aktivasyonu uygular (block içinde)

### 3. Çıkış Katmanı

Residual block'ların çıktısını Q-değerlerine dönüştürür:
```
hidden → W_out @ hidden + b_out → Q-values (4 eylem için)
```

---

## MLP'den Farkları

### Derinlik

- **MLP:** Genellikle 2-3 gizli katman (daha derin olursa vanishing gradient)
- **ResNet:** 3+ residual block ile daha derin ağlar eğitilebilir

### Öğrenme Hızı

- **MLP:** Alt katmanlar yavaş öğrenir (küçük gradient'ler)
- **ResNet:** Tüm katmanlar hızlı öğrenir (skip connection korumalı gradient)

### Eğitim Stabilitesi

- **MLP:** Derin ağlarda eğitim kararsız olabilir
- **ResNet:** Skip connection eğitimi stabilize eder

---

## GridWorld'deki Kullanımı

ResNet, GridWorld problemini çözerken:

- **Karmaşık Özellik Kombinasyonları:** Birden fazla residual block ile, basit özelliklerden (konum) karmaşık stratejilere (optimal yol) kadar öğrenebilir
- **Gradyan Korunumu:** Her block'un gradient'i koruması sayesinde, tüm katmanlar eşit hızda öğrenir
- **Derin Temsil:** Daha derin ağ, daha soyut ve karmaşık özellikleri öğrenebilir (örneğin: "döngüden kaçınma", "optimal yol seçimi")

### Öğrenme Süreci

1. **İlk Block'lar:** Temel özellikleri öğrenir (örneğin: hedefe yönelme)
2. **Orta Block'lar:** Özellik kombinasyonlarını öğrenir (örneğin: mesafe ve konum ilişkisi)
3. **Son Block'lar:** Karmaşık stratejileri öğrenir (örneğin: döngü önleme, optimal yol)


