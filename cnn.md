# CNN Nasıl Çalışır?

---

## Görüntü Formatında Düşünme

MLP, dünyayı sadece iki sayı olarak görür: `(x, y)` koordinatları. CNN ise çok daha zengin bir bakış açısına sahiptir. Dünyayı bir **görüntü** olarak görür, tıpkı bir insanın gözleri gibi!

### GridWorld'i Görüntüye Dönüştürmek

CNN'in ilk yaptığı şey, GridWorld'i 5x5 piksellik bir görüntüye dönüştürmektir. Ama sıradan bir görüntü değil - **3 kanallı** bir görüntü! Her kanal farklı bir bilgi taşır:

- **Kanal 0 - Ajan Kanallı:** Ajanın bulunduğu hücre parlak beyaz (1.0), diğer hücreler siyah (0.0). Sanki bir projektör, ajanın üzerine ışık tutuyor.
- **Kanal 1 - Hedef Kanallı:** Hedefin bulunduğu hücre parlak beyaz (1.0), diğerleri siyah. Bu, "hedef orada!" diye bağıran bir işaret lambası gibi.
- **Kanal 2 - Mesafe Kanallı:** Her hücreden hedefe olan mesafeyi gösteren bir ısı haritası. Hedefe yakın yerler koyu, uzak yerler açık. Bu, ajanın "yön hissini" kazanmasına yardımcı olur.

Bu üç kanal birleştiğinde, CNN için zengin bir bilgi kaynağı oluşur. Artık CNN sadece "ajan (2,1) konumunda" demekle kalmaz; "ajan sağ üstte, hedef sağ altta, aralarında 5 birimlik mesafe var" gibi çok daha detaylı bir anlayışa sahip olur.

### Convolution: Filtreler ile Tarama

MLP'de her nöron tüm görüntüye bakardı. CNN ise farklı bir strateji kullanır: **Küçük filtreler (kernels)** ile görüntüyü tarar. Bu filtreler, tıpkı parmak izi tanıma sistemlerindeki gibi, lokal desenleri yakalar.

Her filtre 3x3'lük küçük bir penceredir. Bu pencere görüntü üzerinde kayarak geçer ve her pozisyonda bir işlem yapar:

```
Filtre:  [w1, w2, w3]
         [w4, w5, w6]
         [w7, w8, w9]

Görüntü Bölgesi: [p1, p2, p3]
                  [p4, p5, p6]
                  [p7, p8, p9]

Çıktı = w1*p1 + w2*p2 + ... + w9*p9 + bias
```

Bu işlem, filtre ile görüntü bölgesinin ne kadar benzer olduğunu ölçer. Eğer filtre "sağ alt köşede bir şey var mı?" diye soruyorsa ve görüntü bölgesinde gerçekten sağ alt köşede bir şey varsa, çıktı büyük bir değer üretir.

**16 farklı filtre** vardır. Her biri farklı bir şey arar:
- Filtre 1: "Ajan sağda, hedef solda mı?" 
- Filtre 2: "Hedef ajanın üstünde mi?"
- Filtre 3: "Mesafe çok mu yakın?"
- ... ve böyle devam eder.

Her filtre, tüm görüntü üzerinde kayarak geçer ve bir "feature map" (özellik haritası) üretir. Bu harita, o filtrenin nerede aktif olduğunu (nerede aradığı deseni bulduğunu) gösterir.

### ReLU: Negatif Değerleri Sıfırlama

Convolution işleminden sonra ReLU gelir. ReLU, negatif değerleri sıfıra çevirir. Neden?

Sezgisel olarak: "Eğer bir filtre, aradığı deseni bulamadıysa (negatif veya sıfır çıktı), bu bilgi önemsizdir. Bir sonraki katmana aktarmaya gerek yok."

ReLU sayesinde CNN, sadece **önemli bilgileri** bir sonraki katmana aktarır. Bu, hesaplama verimliliğini artırır ve ağın daha hızlı öğrenmesini sağlar.

### MaxPooling: Görüntüyü Küçültme

MaxPooling, görüntüyü küçültür. 2x2'lik bölgelerden maksimum değeri alır ve görüntüyü yarıya indirir.

Neden yapıyoruz bunu?

1. **Hesaplama Tasarrufu:** Daha küçük görüntü = daha az işlem = daha hızlı öğrenme.
2. **Genelleme:** MaxPooling, "bu bölgede bir şey var mı?" gibi daha genel sorular sorar. Detaylar kaybolur ama önemli bilgiler korunur.
3. **Konum Bağımsızlığı:** Ajan hafifçe sağa kayarsa, MaxPooling sayesinde aynı özellik hala yakalanır.

Örnek: 5x5 görüntü → MaxPool → 2x2 görüntü. Artık CNN, "sol üstte bir şey var, sağ altta bir şey var" gibi daha genel desenler görebilir.

### İkinci Convolution: Daha Karmaşık Desenler

İlk convolution'dan sonra MaxPool gelir, sonra **ikinci bir convolution** katmanı daha gelir. Bu sefer 32 filtre vardır ve daha karmaşık desenler arar:

- İlk convolution: "Ajan ve hedef nerede?" (basit desenler)
- İkinci convolution: "Ajan ve hedef arasındaki ilişki nedir?" (karmaşık desenler)

Bu katmanlı yapı sayesinde CNN, önce basit özellikleri öğrenir, sonra bunları birleştirerek karmaşık stratejiler oluşturur.

---

## MLP'den Farkı

### Uzamsal Farkındalık

MLP, `(x, y)` koordinatlarını alır ve bunları doğrudan işler. Ama bu koordinatlar **uzamsal ilişkileri** kaybetmişlerdir. CNN ise görüntü formatında çalıştığı için:

- "Ajan hedefin sağında mı, solunda mı?" sorusunu otomatik olarak cevaplayabilir.
- Komşu hücreler arasındaki ilişkiyi öğrenebilir.
- Geometrik desenleri (örn: "diyagonal bir yol var") yakalayabilir.

### Parametre Verimliliği

MLP'de her nöron tüm görüntüye bağlıdır. 5x5x3 = 75 girdi için, 64 nöronlu bir katman 75 × 64 = 4800 parametre gerektirir.

CNN'de ise her filtre sadece 3x3 = 9 parametreye sahiptir ve **tüm görüntüye uygulanır**. 16 filtre için toplam 16 × 9 = 144 parametre (bias'lar hariç). Bu, **çok daha verimli** bir öğrenmedir!

### Translasyon Invaryantlığı

CNN, görüntüdeki desenlerin **nerede** olduğuna değil, **ne olduğuna** odaklanır. Örneğin, "ajan hedefin üstünde" deseni öğrenmişse, bu desen görüntünün hangi bölgesinde olursa olsun tanıyabilir. MLP bunu yapamaz çünkü her konum farklı bir girdi nöronuna karşılık gelir.

---

## Öğrenme Süreci

### İlk Adımlar: Keşif

Başlangıçta filtreler rastgele ağırlıklara sahiptir. Henüz hiçbir şey bilmiyorlar. Ajan rastgele hareket eder ve deneyimler toplar.

### Filtrelerin Uzmanlaşması

Eğitim ilerledikçe, her filtre kendi uzmanlık alanını geliştirir:
- Bazı filtreler "hedef yakınlığı" öğrenir.
- Bazıları "duvar yakınlığı" öğrenir.
- Bazıları "yön bilgisi" öğrenir.

Bu uzmanlaşma, geri yayılım (backpropagation) sayesinde olur. Her hata, filtrelerin ağırlıklarını günceller ve bir sonraki seferde daha iyi performans gösterirler.

### Feature Map'lerin Anlamı

Eğitim sonrası feature map'lere baktığımızda:
- Bazı feature map'ler ajanın pozisyonunda parlıyordur.
- Bazıları hedefin pozisyonunda parlıyordur.
- Bazıları mesafe bilgisini görselleştiriyordur.

Bu, CNN'in "düşündüğü şeyi görmemize" izin verir. Sanki CNN'in zihnini okuyormuş gibi!

---

## Özet

CNN, GridWorld problemini çözerken:

- **Görüntü formatında düşünür:** Dünyayı zengin bir temsil olarak görür.
- **Lokal desenleri yakalar:** Küçük filtrelerle uzamsal ilişkileri öğrenir.
- **Verimli öğrenir:** Parametre paylaşımı sayesinde az veri ile çok şey öğrenir.
- **Genelleme yapar:** MaxPooling sayesinde farklı pozisyonlarda aynı stratejiyi uygular.

MLP gibi "tüm görüntüye aynı anda bakar ve her şeyi karıştırır" demez. CNN, bir dedektif gibi, küçük ipuçlarını yakalar, bunları birleştirir ve sonunda "ajan hedefe ulaşmak için sağa gitmeli" gibi bir sonuca varır.

Umarım bu yolculuk, bir CNN'in zihninde neler olup bittiğini ve neden görüntü formatını tercih ettiğini daha sezgisel bir şekilde anlamana yardımcı olmuştur!

