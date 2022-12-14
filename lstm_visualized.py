
from __future__ import annotations
# -*- coding: utf-8 -*-
"""
LSTM'in Karar Verme Sürecinin Pygame ile Görselleştirilmesi
---------------------------------------------------------
Bu dosya, eğitilmiş bir LSTM-DQN ajanının her bir adımda sinir ağı içinde
neler olduğunu görsel olarak anlamak için tasarlanmıştır.

- Geçmiş adımlar nasıl sequence olarak giriyor?
- LSTM'in hidden state ve cell state'i nasıl değişiyor?
- Gate'ler (forget, input, output) nasıl çalışıyor?
- Geçmiş bilgi nasıl hatırlanıyor ve unutuluyor?
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

# lstm_numpy.py dosyasından import edilen sınıflar ve fonksiyonlar
from lstm_numpy import (
    LSTM,
    train_lstm_dqn
)
from common import GridWorld, to_state_vector

# Pygame Görselleştirme Kodları
# ---------------------------------------------------------------------

# Renkler ve Ayarlar
BG_COLOR = (10, 10, 40)
GRID_COLOR = (80, 80, 120)
AGENT_COLOR = (255, 200, 0)
GOAL_COLOR = (0, 255, 150)
TEXT_COLOR = (240, 240, 240)
CHOSEN_ACTION_COLOR = (0, 255, 150)
GATE_COLOR = (255, 150, 100)
CELL_STATE_COLOR = (150, 200, 255)
HIDDEN_STATE_COLOR = (200, 150, 255)

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

def draw_sequence(screen, sequence, x_offset, y_offset, title, scale=30, font=None):
    """ Sequence'i (zaman içinde durumları) çizer. """
    if font is None:
        font = pygame.font.SysFont("monospace", 12)
    title_surf = font.render(title, True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset - 20))
    
    # Her zaman adımı için
    for t, state_vec in enumerate(sequence):
        y_pos = y_offset + t * (scale + 5)
        
        # Zaman etiketi
        time_label = f"t-{len(sequence)-1-t}"
        time_surf = font.render(time_label, True, TEXT_COLOR)
        screen.blit(time_surf, (x_offset, y_pos))
        
        # Durum değerlerini çiz (x, y)
        x_val, y_val = state_vec[0], state_vec[1]
        
        # X değeri
        x_text = f"x:{x_val:.2f}"
        x_surf = font.render(x_text, True, TEXT_COLOR)
        screen.blit(x_surf, (x_offset + 50, y_pos))
        
        # Y değeri
        y_text = f"y:{y_val:.2f}"
        y_surf = font.render(y_text, True, TEXT_COLOR)
        screen.blit(y_surf, (x_offset + 120, y_pos))
        
        # Görsel gösterim (renkli kare)
        color_x = (int(255 * x_val), 0, 0)
        color_y = (0, int(255 * y_val), 0)
        pygame.draw.rect(screen, color_x, (x_offset + 180, y_pos, 15, 15))
        pygame.draw.rect(screen, color_y, (x_offset + 200, y_pos, 15, 15))

def draw_gates(screen, cache, x_offset, y_offset, title, font):
    """ Gate aktivasyonlarını çizer. """
    if cache is None:
        return
    
    small_font = pygame.font.SysFont("monospace", 12)
    title_surf = small_font.render(title, True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset))
    
    # Son zaman adımının gate'lerini göster
    f_t = cache['f_t'][0]  # Forget gate
    i_t = cache['i_t'][0]  # Input gate
    o_t = cache['o_t'][0]  # Output gate
    
    # Her gate için ortalama değeri göster (tüm hidden_dim için)
    gate_mean_f = f_t.mean()
    gate_mean_i = i_t.mean()
    gate_mean_o = o_t.mean()
    
    y_pos = y_offset + 25
    
    # Forget Gate
    forget_text = f"Forget: {gate_mean_f:.3f}"
    forget_surf = small_font.render(forget_text, True, GATE_COLOR)
    screen.blit(forget_surf, (x_offset, y_pos))
    
    # Bar chart
    bar_width = int(gate_mean_f * 100)
    pygame.draw.rect(screen, GATE_COLOR, (x_offset + 100, y_pos + 2, bar_width, 15))
    
    # Input Gate
    input_text = f"Input: {gate_mean_i:.3f}"
    input_surf = small_font.render(input_text, True, GATE_COLOR)
    screen.blit(input_surf, (x_offset, y_pos + 25))
    bar_width = int(gate_mean_i * 100)
    pygame.draw.rect(screen, GATE_COLOR, (x_offset + 100, y_pos + 27, bar_width, 15))
    
    # Output Gate
    output_text = f"Output: {gate_mean_o:.3f}"
    output_surf = small_font.render(output_text, True, GATE_COLOR)
    screen.blit(output_surf, (x_offset, y_pos + 50))
    bar_width = int(gate_mean_o * 100)
    pygame.draw.rect(screen, GATE_COLOR, (x_offset + 100, y_pos + 52, bar_width, 15))

def draw_state_vectors(screen, h_t, c_t, x_offset, y_offset, font):
    """ Hidden state ve cell state'i çizer. """
    if h_t is None or c_t is None:
        return
    
    small_font = pygame.font.SysFont("monospace", 12)
    
    # Hidden State
    h_mean = h_t[0].mean()
    h_std = h_t[0].std()
    h_text = f"Hidden State: μ={h_mean:.3f}, σ={h_std:.3f}"
    h_surf = small_font.render(h_text, True, HIDDEN_STATE_COLOR)
    screen.blit(h_surf, (x_offset, y_offset))
    
    # Hidden state histogram (basitleştirilmiş)
    y_pos = y_offset + 20
    sample_h = h_t[0][:10]  # İlk 10 değeri göster
    for i, val in enumerate(sample_h):
        normalized = (val + 1) / 2  # -1..1 -> 0..1
        bar_width = int(normalized * 50)
        pygame.draw.rect(screen, HIDDEN_STATE_COLOR, (x_offset + i * 6, y_pos, 5, bar_width))
    
    # Cell State
    c_mean = c_t[0].mean()
    c_std = c_t[0].std()
    c_text = f"Cell State: μ={c_mean:.3f}, σ={c_std:.3f}"
    c_surf = small_font.render(c_text, True, CELL_STATE_COLOR)
    screen.blit(c_surf, (x_offset, y_offset + 50))
    
    # Cell state histogram
    y_pos = y_offset + 70
    sample_c = c_t[0][:10]
    for i, val in enumerate(sample_c):
        normalized = (val + 1) / 2
        bar_width = int(normalized * 50)
        pygame.draw.rect(screen, CELL_STATE_COLOR, (x_offset + i * 6, y_pos, 5, bar_width))

def draw_q_values(screen, q_values, chosen_action, x_offset, y_offset, font):
    """ Q-değerlerini çizer. """
    action_labels = ["↑", "↓", "←", "→"]
    action_names = ["Yukarı", "Aşağı", "Sol", "Sağ"]
    
    small_font = pygame.font.SysFont("monospace", 14)
    title_surf = small_font.render("Q-Değerleri:", True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset))
    
    for i, (label, name, q_val) in enumerate(zip(action_labels, action_names, q_values)):
        y_pos = y_offset + 25 + i * 25
        color = CHOSEN_ACTION_COLOR if i == chosen_action else TEXT_COLOR
        
        text = f"{label} {name}: {q_val:.2f}"
        text_surf = small_font.render(text, True, color)
        screen.blit(text_surf, (x_offset, y_pos))
        
        bar_width = int(abs(q_val) * 15)
        bar_color = CHOSEN_ACTION_COLOR if i == chosen_action else (150, 150, 150)
        pygame.draw.rect(screen, bar_color, (x_offset + 120, y_pos + 5, bar_width, 15))

def main():
    # 1. Adım: Ajanı eğit
    env, policy_net, _, _ = train_lstm_dqn()
    s = env.reset()
    
    sequence_length = 10
    sequence = []

    # 2. Adım: Pygame'i başlat
    pygame.init()
    screen_width, screen_height = 1400, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("LSTM-DQN Görselleştirme")
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

        # Sequence'e durum ekle
        s_vec = to_state_vector(s, env.w, env.h)
        sequence.append(s_vec)
        
        # Sequence'i doldur
        if len(sequence) < sequence_length:
            seq_array = (sequence * (sequence_length // len(sequence) + 1))[:sequence_length]
        else:
            seq_array = sequence[-sequence_length:]
        
        seq_array = np.array(seq_array)
        
        # İleri yayılım
        q_values, all_cache = policy_net.forward(seq_array[None, :, :])
        q_values = q_values[0]  # Batch boyutunu kaldır
        
        # Son zaman adımının cache'ini al
        last_cache = all_cache[-1]
        h_t = last_cache['h_t']
        c_t = last_cache['c_t']
        
        action = int(np.argmax(q_values))

        # Çizim işlemleri
        screen.fill(BG_COLOR)
        
        # Sol üst: GridWorld haritası
        draw_grid(screen, env, cell_size)
        
        # Sağ üst: Sequence (geçmiş durumlar)
        draw_sequence(screen, seq_array, 400, 50, "Geçmiş Durumlar (Sequence)", scale=25, font=small_font)
        
        # Orta sağ: Gate'ler
        draw_gates(screen, last_cache, 700, 50, "Gate Aktivasyonları", small_font)
        
        # Orta alt: Hidden ve Cell State
        draw_state_vectors(screen, h_t, c_t, 700, 250, small_font)
        
        # Sol alt: Q-değerleri
        draw_q_values(screen, q_values, action, 10, grid_height + 80, font)
        
        # Bilgi metinlerini yazdır
        state_text = f"Durum (x,y): {s}"
        text1 = font.render(state_text, True, TEXT_COLOR)
        screen.blit(text1, (10, screen_height - 60))
        
        action_text = f"Seçilen Eylem: {['↑ Yukarı', '↓ Aşağı', '← Sol', '→ Sağ'][action]}"
        text2 = font.render(action_text, True, CHOSEN_ACTION_COLOR)
        screen.blit(text2, (10, screen_height - 35))
        
        seq_text = f"Sequence Uzunluğu: {len(sequence)}/{sequence_length}"
        text3 = small_font.render(seq_text, True, TEXT_COLOR)
        screen.blit(text3, (10, screen_height - 10))

        pygame.display.flip()

        # Bir sonraki adıma geç
        s, _, done, _ = env.step(action)
        if done:
            s = env.reset()
            sequence = []  # Sequence'i sıfırla
            pygame.time.wait(1000)

        # Animasyon hızını ayarla
        clock.tick(2)  # Saniyede 2 adım

    pygame.quit()

if __name__ == "__main__":
    main()

