# ResNet Nasıl Çalışır?

---

## Skip Connection'lar

MLP, derin ağlarda gradyanlar küçüldüğü için (vanishing gradient) zorlanır. ResNet, **skip connection** (atlamalı bağlantı) kullanarak bu sorunu çözer.

### Residual Block

Residual block, şu formülü kullanır:
```
output = F(x) + x
```

Burada:
- `F(x)`: İki katmanlı bir transformasyon (öğrenilen değişiklik)
- `x`: Girdi (skip connection ile direkt geçer)
- `output`: Transformasyon + orijinal girdi

### Skip Connection'ın Faydası

Skip connection, gradyanların direkt geçmesine izin verir:
- Normal ağda: Gradyan her katmandan geçerken küçülür
- ResNet'te: Gradyan skip connection'dan direkt geçer, korunur

Bu sayede daha derin ağlar eğitilebilir hale gelir.

---

## Özet

ResNet, GridWorld problemini çözerken:
- **Gradyan akışını iyileştirir:** Skip connection'lar gradyanları korur
- **Derin ağları eğitir:** 3+ katmanlı ağları kolayca eğitir
- **Öğrenmeyi hızlandırır:** Identity mapping sayesinde hızlı öğrenir

Umarım bu yolculuk, ResNet'in zihninde neler olup bittiğini daha sezgisel bir şekilde anlamana yardımcı olmuştur!

