
from __future__ import annotations
# -*- coding: utf-8 -*-
"""
CNN'in Karar Verme Sürecinin Pygame ile Görselleştirilmesi
---------------------------------------------------------
Bu dosya, eğitilmiş bir CNN-DQN ajanının her bir adımda sinir ağı içinde
neler olduğunu görsel olarak anlamak için tasarlanmıştır.

- GridWorld nasıl görüntü formatına çevriliyor? (3 kanal: ajan, hedef, mesafe)
- Convolution filtreleri ne öğreniyor?
- Feature map'ler nasıl görünüyor?
- MaxPooling nasıl örnekleme yapıyor?
- Q-değerleri nasıl üretiliyor?

Bu soruların cevaplarını animasyonlu bir şekilde gösterir.

Çalıştırmak için:
1. Gerekliyse `pip install pygame` ile kütüphaneyi yükleyin.
2. Bu dosyayı normal bir Python scripti olarak çalıştırın.
3. Önce modelin eğitilmesi için konsolu takip edin.
4. Eğitim bitince Pygame penceresi açılacaktır.
"""
import pygame
import numpy as np

# cnn_numpy.py dosyasından import edilen sınıflar ve fonksiyonlar
from cnn_numpy import (
    CNN,
    to_image_grid,
    train_cnn_dqn
)
from common import GridWorld

# Pygame Görselleştirme Kodları
# ---------------------------------------------------------------------

# Renkler ve Ayarlar
BG_COLOR = (10, 10, 40)
GRID_COLOR = (80, 80, 120)
AGENT_COLOR = (255, 200, 0)
GOAL_COLOR = (0, 255, 150)
TEXT_COLOR = (240, 240, 240)
CHOSEN_ACTION_COLOR = (0, 255, 150)
FEATURE_MAP_COLOR = (100, 150, 255)

def draw_grid(screen, env, cell_size):
    """ GridWorld haritasını çizer. """
    for y in range(env.h):
        for x in range(env.w):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
    goal_x, goal_y = env.goal
    goal_rect = pygame.Rect(goal_x * cell_size + 5, goal_y * cell_size + 5, cell_size - 10, cell_size - 10)
    pygame.draw.rect(screen, GOAL_COLOR, goal_rect)
    agent_x, agent_y = env.state
    pygame.draw.circle(screen, AGENT_COLOR, (int(agent_x * cell_size + cell_size / 2), int(agent_y * cell_size + cell_size / 2)), int(cell_size / 3))

def draw_image_grid(screen, img, x_offset, y_offset, title, scale=40):
    """
    Görüntüyü (5x5x3) çizer.
    Her kanal ayrı bir kare olarak gösterilir.
    """
    h, w, c = img.shape
    small_font = pygame.font.SysFont("monospace", 12)
    title_surf = small_font.render(title, True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset - 20))
    
    # Her kanal için
    for chan in range(c):
        chan_label = ["Ajan", "Hedef", "Mesafe"][chan]
        label_surf = small_font.render(chan_label, True, TEXT_COLOR)
        screen.blit(label_surf, (x_offset + chan * (w * scale + 10), y_offset - 10))
        
        for i in range(h):
            for j in range(w):
                val = img[i, j, chan]
                # Değeri 0-255 arasına ölçeklendir
                color_val = int(255 * val)
                color = (color_val, color_val, color_val)
                rect = pygame.Rect(x_offset + chan * (w * scale + 10) + j * scale,
                                   y_offset + i * scale,
                                   scale - 2, scale - 2)
                pygame.draw.rect(screen, color, rect)

def draw_feature_map(screen, feature_map, x_offset, y_offset, title, scale=15):
    """
    Feature map'i (convolution çıktısı) çizer.
    Her filtre için bir mini görüntü gösterilir.
    """
    batch, h, w, num_filters = feature_map.shape
    fm = feature_map[0]  # İlk batch elemanı
    
    small_font = pygame.font.SysFont("monospace", 12)
    title_surf = small_font.render(title, True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset - 20))
    
    # En fazla 8 filtreyi göster (görsel karmaşıklığı azaltmak için)
    num_to_show = min(8, num_filters)
    filters_per_row = 4
    
    for idx in range(num_to_show):
        row = idx // filters_per_row
        col = idx % filters_per_row
        
        # Normalize et
        fm_slice = fm[:, :, idx]
        fm_min, fm_max = fm_slice.min(), fm_slice.max()
        if fm_max > fm_min:
            fm_slice = (fm_slice - fm_min) / (fm_max - fm_min)
        
        # Çiz
        for i in range(h):
            for j in range(w):
                val = fm_slice[i, j]
                color_val = int(255 * val)
                color = (color_val, color_val, color_val)
                rect = pygame.Rect(x_offset + col * (w * scale + 5) + j * scale,
                                   y_offset + row * (h * scale + 5) + i * scale,
                                   scale - 1, scale - 1)
                pygame.draw.rect(screen, color, rect)

def draw_q_values(screen, q_values, chosen_action, x_offset, y_offset, font):
    """ Q-değerlerini çizer. """
    action_labels = ["↑", "↓", "←", "→"]
    action_names = ["Yukarı", "Aşağı", "Sol", "Sağ"]
    
    small_font = pygame.font.SysFont("monospace", 14)
    title_surf = small_font.render("Q-Değerleri:", True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset))
    
    for i, (label, name, q_val) in enumerate(zip(action_labels, action_names, q_values)):
        y_pos = y_offset + 30 + i * 25
        color = CHOSEN_ACTION_COLOR if i == chosen_action else TEXT_COLOR
        
        # Eylem adı ve değeri
        text = f"{label} {name}: {q_val:.2f}"
        text_surf = small_font.render(text, True, color)
        screen.blit(text_surf, (x_offset, y_pos))
        
        # Değer çubuğu (görsel gösterim)
        bar_width = int(abs(q_val) * 20)
        bar_color = CHOSEN_ACTION_COLOR if i == chosen_action else (150, 150, 150)
        pygame.draw.rect(screen, bar_color, (x_offset + 120, y_pos + 5, bar_width, 15))

def main():
    # 1. Adım: Ajanı eğit
    env, policy_net, _, _ = train_cnn_dqn()
    s = env.reset()

    # 2. Adım: Pygame'i başlat
    pygame.init()
    screen_width, screen_height = 1400, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("CNN-DQN Görselleştirme")
    font = pygame.font.SysFont("monospace", 16, bold=True)
    small_font = pygame.font.SysFont("monospace", 14)

    cell_size = 60
    grid_width = cell_size * env.w
    grid_height = cell_size * env.h

    # 3. Adım: Ana döngüyü başlat
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Ajanın karar verme süreci
        img = to_image_grid(s, env.w, env.h, env.goal)
        q_values, cache = policy_net.forward(img[None, :, :, :])
        
        # Cache'den ara değerleri çıkar
        X, conv1_out, conv1_act, pool1_out, conv2_out, conv2_act, pool2_out, _, _, _ = cache
        q_values = q_values[0]  # Batch boyutunu kaldır
        action = int(np.argmax(q_values))

        # Çizim işlemleri
        screen.fill(BG_COLOR)
        
        # Sol üst: GridWorld haritası
        draw_grid(screen, env, cell_size)
        
        # Sağ üst: Girdi görüntüsü (3 kanal)
        draw_image_grid(screen, img, 400, 50, "Girdi Görüntüsü", scale=40)
        
        # Orta sağ: Conv1 feature map'leri
        draw_feature_map(screen, conv1_act, 650, 50, "Conv1 Özellikleri", scale=12)
        
        # Orta alt: Conv2 feature map'leri
        draw_feature_map(screen, conv2_act, 650, 300, "Conv2 Özellikleri", scale=8)
        
        # Sol alt: Q-değerleri
        draw_q_values(screen, q_values, action, 10, grid_height + 80, font)
        
        # Bilgi metinlerini yazdır
        state_text = f"Durum (x,y): {s}"
        text1 = font.render(state_text, True, TEXT_COLOR)
        screen.blit(text1, (10, screen_height - 60))
        
        action_text = f"Seçilen Eylem: {['↑ Yukarı', '↓ Aşağı', '← Sol', '→ Sağ'][action]}"
        text2 = font.render(action_text, True, CHOSEN_ACTION_COLOR)
        screen.blit(text2, (10, screen_height - 35))

        pygame.display.flip()

        # Bir sonraki adıma geç
        s, _, done, _ = env.step(action)
        if done:
            s = env.reset()
            pygame.time.wait(1000)  # Hedefe ulaşınca bekle

        # Animasyon hızını ayarla
        clock.tick(2)  # Saniyede 2 adım

    pygame.quit()

if __name__ == "__main__":
    main()

