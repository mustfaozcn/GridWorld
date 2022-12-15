# Attention Mekanizması

---

## Hangi Bilgi Önemli?

MLP, CNN ve LSTM, tüm bilgilere eşit önem verir. Ama Attention farklıdır: **Hangi bilgi önemli?** sorusunu sorar ve ona göre odaklanır.

Attention, tıpkı bir insanın bir fotoğrafa baktığında önce önemli kısımlara (yüzler, objeler) bakması gibi, ağın da hangi özelliklerin daha önemli olduğunu öğrenmesine yardımcı olur.

### Özellik Vektörü

Attention için durum, zengin bir özellik vektörü olarak temsil edilir:
- Ajanın konumu (x, y)
- Hedefin konumu (goal_x, goal_y)
- Ajan-hedef mesafesi
- Duvarlara uzaklık

Bu özellikler, attention'ın hangi bilgilerin önemli olduğunu öğrenmesine yardımcı olur.

### Query, Key, Value

Attention mekanizması üç temel bileşen kullanır:

- **Query (Sorgu):** "Ne arıyorum?" - Ağın ilgilendiği bilgi
- **Key (Anahtar):** "Ne sunuyorum?" - Her özelliğin ne tür bilgi içerdiği
- **Value (Değer):** "Gerçek bilgi" - Her özelliğin gerçek değeri

Attention, Query ile Key'leri karşılaştırarak hangi Value'ların önemli olduğunu belirler.

### Attention Weight'leri

Attention weight'leri, her özelliğe ne kadar "dikkat" verileceğini gösterir. 0 ile 1 arasında değerler alır:
- 0: "Bu özellik önemsiz"
- 1: "Bu özellik çok önemli"

Örnek: Eğer ajan hedefe çok yakınsa, "mesafe" özelliğine yüksek attention weight verilir.

### Multi-Head Attention

Multi-head attention, aynı bilgiyi farklı açılardan analiz eder:
- Head 1: "Ajan-hedef ilişkisi"ne odaklanır
- Head 2: "Duvarlara yakınlık"a odaklanır
- Head 3: "Konum bilgisi"ne odaklanır
- ...

Bu sayede ağ, bilgiyi çok yönlü analiz edebilir.

---

## Çalışma Mantığı

### Scaled Dot-Product Attention

Attention'ın kalbi, şu formülü kullanır:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Bu formül:
1. Query ve Key'leri çarparak benzerlik skorları hesaplar
2. Softmax ile attention weight'lerini normalize eder
3. Bu weight'lerle Value'ları ağırlıklandırır

### Self-Attention: Özellikler Arası İlişkiler

Self-attention, özelliklerin birbiriyle nasıl ilişkili olduğunu öğrenir:
- "Ajan x koordinatı" ile "Hedef x koordinatı" arasındaki ilişki
- "Mesafe" ile "Duvarlara uzaklık" arasındaki ilişki
- ...

Bu sayede ağ, kompleks özellik kombinasyonlarını öğrenebilir.

---

## Özet

Attention, GridWorld problemini çözerken:

- **Önemli bilgiyi vurgular:** Hangi özelliklerin daha önemli olduğunu öğrenir
- **Önemsiz bilgiyi bastırır:** Gürültülü veya gereksiz bilgileri görmezden gelir
- **Özellik ilişkilerini öğrenir:** Self-attention ile özellikler arası ilişkileri yakalar
- **Çok yönlü analiz:** Multi-head attention ile farklı açılardan bilgiyi analiz eder

MLP gibi "tüm bilgilere eşit önem verir" demez. CNN gibi "lokal desenlere odaklanır" demez. LSTM gibi "geçmişi hatırlar" demez. Attention, **"şu an hangi bilgi önemli?"** diye sorar ve ona göre odaklanır.


