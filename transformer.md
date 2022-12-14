# Transformer Nasıl Çalışır?

---

## Self-Attention ve Positional Encoding

Transformer, self-attention ve positional encoding kullanarak özellikler arası kompleks ilişkileri öğrenir.

### Self-Attention

Self-attention, özelliklerin birbiriyle nasıl ilişkili olduğunu öğrenir:
- "Agent X" ile "Goal X" arasındaki ilişki
- "Distance" ile diğer özellikler arasındaki ilişki
- ...

### Positional Encoding

Positional encoding, her özelliğin konumunu kodlar. Bu sayede ağ, hangi özelliğin nerede olduğunu öğrenir.

### Multi-Head Attention

Farklı head'ler farklı açılardan bilgiyi analiz eder.

---

## Özet

Transformer, GridWorld problemini çözerken:
- **Self-attention ile özellik ilişkilerini öğrenir**
- **Positional encoding ile konum bilgisini kodlar**
- **Multi-head attention ile çok yönlü analiz yapar**

Umarım bu yolculuk, Transformer'ın zihninde neler olup bittiğini daha sezgisel bir şekilde anlamana yardımcı olmuştur!

