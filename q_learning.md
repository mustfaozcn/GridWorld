# Q-Learning Nasıl Çalışır?

Bu doküman, `q_learning.py` dosyasındaki Q-Learning algoritmasının çalışma prensiplerini açıklar. DQN'den farklı olarak Q-Learning, sinir ağı yerine tablo tabanlı (tabular) bir yaklaşım kullanır.

DQN, bir sinir ağı ile Q-değerlerini öğrenirken, Q-Learning her durum-eylem çifti için Q-değerlerini doğrudan bir tabloda saklar. Bu yaklaşım, küçük state space'lerde son derece etkilidir.

---

## Q-Tablosu

Q-Tablosu, her durum-eylem çifti için beklenen toplam ödülü (Q-değeri) saklayan bir veri yapısıdır. NumPy dizisi olarak şu şekilde temsil edilir:

```python
Q = np.full((env.h, env.w, 4), ...)
```

Boyutlar:
- `env.h × env.w`: GridWorld'ün her hücresi bir durumdur
- `4`: Her durumda 4 olası eylem (yukarı, aşağı, sol, sağ)

Tablonun her elemanı `Q[y, x, a]`, `(x, y)` konumunda `a` eylemini yapmanın beklenen toplam ödülünü ifade eder.

### Tabular Yaklaşımın Özellikleri

- **Direkt Erişim:** Her durum-eylem çifti için bir değer saklanır, hesaplama yapılmaz
- **Kesin Değerler:** Sinir ağı tahmini yerine, öğrenilen kesin değerler kullanılır
- **Küçük State Space:** Sadece küçük ve ayrık state space'ler için uygundur

Bu yaklaşım, **Tabular Q-Learning** olarak adlandırılır çünkü tüm Q-değerleri bir tabloda saklanır.

---

## Q-Tablosu Nasıl Öğrenir?

Q-tablosu, Bellman denklemi kullanılarak iteratif olarak güncellenir. Öğrenme sürecinin temeli:

```python
Q[y, x, a] += alpha * td_error
```

Bu formülü parçalara ayıralım:

-   **`Q[y, x, a]`:** Tablodaki mevcut değer, `(x, y)` konumunda `a` eylemini yapmanın öğrenilmiş Q-değeri
-   `+=`: "Eski puanı, birazdan hesaplayacağımız yeni bilgiyle güncelle."
-   **`alpha` (Öğrenme Oranı):** Yeni bilgiye ne kadar güvenileceğini kontrol eder
    - Küçük `alpha` (örn: 0.01): Yavaş ve tutucu öğrenme, eski bilgilere daha çok güven
    - Büyük `alpha` (örn: 0.5): Hızlı öğrenme, yeni bilgilere daha çok önem
-   **`td_error` (Temporal Difference Error):** Mevcut tahmin ile hedef arasındaki fark

### Temporal Difference Error (TD Error)

TD error, mevcut Q-değeri tahmini ile gerçek değer arasındaki farktır:

```
td_error = td_target - Q[y, x, a]
```

**TD Target Hesaplama:**

TD target, Bellman denkleminden türetilir ve şu şekilde hesaplanır:

```
td_target = r + gamma * np.max(Q[ny, nx, :])
```

Bileşenler:
- **`r`:** Anlık ödül (eylem sonrası alınan ödül, genellikle -1 veya +10)
- **`gamma`:** Discount faktörü (0-1 arası). Gelecekteki ödüllere ne kadar önem verildiğini kontrol eder
  - `gamma = 0.9`: Gelecekteki ödüllere yüksek önem
  - `gamma = 0.1`: Sadece yakın ödüllere önem
- **`np.max(Q[ny, nx, :])`:** Yeni konumda (`nx, ny`) tüm eylemler arasından en yüksek Q-değeri (optimal eylemin değeri)

**Öğrenme Süreci:**

1. **Mevcut Tahmin:** Ajan `Q[y, x, a]` değerini mevcut tahmin olarak kullanır
2. **Eylem Gerçekleştirme:** `a` eylemi yapılır, `r` ödülü alınır ve yeni konum `(nx, ny)` elde edilir
3. **Hedef Hesaplama:** Bellman denklemine göre, bu eylemin gerçek değeri `td_target` olmalıdır
4. **Hata Hesaplama:** Mevcut tahmin ile hedef arasındaki fark (`td_error`) hesaplanır
5. **Güncelleme:** Q-tablosu, öğrenme oranı (`alpha`) ile güncellenir: `Q[y, x, a] += alpha * td_error`

Bu süreç binlerce iterasyon boyunca tekrarlandığında, Q-tablosundaki değerler optimal Q-değerlerine yakınsar ve ajan optimal politikayı öğrenir.

---

## Exploration-Exploitation Dengesi

Q-Learning'de, ajan her zaman öğrendiği en iyi eylemi seçerse (exploitation), daha iyi alternatifleri keşfedemez (exploration). Bu sorunu çözmek için **epsilon-greedy** stratejisi kullanılır.

### Epsilon-Greedy Stratejisi

- **`epsilon` olasılıkla:** Rastgele eylem seçilir (exploration)
  - Ajan, Q-tablosuna bakmaksızın rastgele bir eylem yapar
  - Bu, yeni durumları ve eylemleri keşfetmesini sağlar
- **`1-epsilon` olasılıkla:** En yüksek Q-değerli eylem seçilir (exploitation)
  - Ajan, mevcut bilgisine göre en iyi eylemi seçer
  - Öğrenilen politikanın performansını optimize eder

### Epsilon Decay

Eğitim sürecinde epsilon zamanla azalır (decay):
- **Başlangıçta:** `epsilon = 1.0` (tamamen rastgele)
- **Eğitim ilerledikçe:** `epsilon` azalır (örn: `epsilon = max(0.1, 1.0 - episode/1000)`)
- **Sonunda:** `epsilon ≈ 0.1` (çoğunlukla öğrenilen politikayı kullan)

Bu yaklaşım sayesinde ajan:
- İlk aşamada ortamı keşfeder
- Öğrendikçe öğrenilen politikayı uygular
- Ancak küçük bir olasılıkla yeni yollar keşfetmeye devam eder
