
# -*- coding: utf-8 -*-
"""
Tabular Q-Learning Karar Sürecinin Pygame ile Görselleştirilmesi
----------------------------------------------------------------
Bu dosya, eğitilmiş bir Tabular Q-Learning ajanının her adımda
Q-Tablosu'na bakarak nasıl karar verdiğini görsel olarak anlamak
için tasarlanmıştır.

- Ajan haritadaki konumuna göre Q-Tablosu'nda nereye bakıyor?
- O konumdaki 4 eylemin Q-değerleri neler?
- En iyi eylemi (en yüksek Q-değerini) nasıl seçiyor?

Bu soruların cevaplarını animasyonlu bir şekilde gösterir.

Çalıştırmak için:
1. Gerekliyse `pip install pygame numpy` ile kütüphaneleri yükleyin.
2. Bu dosyayı normal bir Python scripti olarak çalıştırın.
3. Önce modelin eğitilmesi için konsolu takip edin.
4. Eğitim bitince Pygame penceresi açılacaktır.
"""
from __future__ import annotations
import pygame
import numpy as np

# q_learning.py dosyasından import edilen sınıflar ve fonksiyonlar
from q_learning import train
from common import GridWorld


# Pygame Görselleştirme Kodları
# ---------------------------------------------------------------------

# Renkler ve Ayarlar
BG_COLOR = (40, 10, 10)
GRID_COLOR = (120, 80, 80)
AGENT_COLOR = (255, 200, 0)
GOAL_COLOR = (0, 255, 150)
TEXT_COLOR = (240, 240, 240)
HIGHLIGHT_COLOR = (255, 255, 0)
CHOSEN_ACTION_COLOR = (0, 255, 150)
INFO_PANEL_COLOR = (50, 20, 20)

def draw_grid_panel(screen, env, cell_size):
    """ Sol tarafta GridWorld haritasını çizer. """
    for y in range(env.h):
        for x in range(env.w):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
    goal_x, goal_y = env.goal
    goal_rect = pygame.Rect(goal_x * cell_size + 5, goal_y * cell_size + 5, cell_size - 10, cell_size - 10)
    pygame.draw.rect(screen, GOAL_COLOR, goal_rect)
    agent_x, agent_y = env.state
    pygame.draw.circle(screen, AGENT_COLOR, (int(agent_x * cell_size + cell_size / 2), int(agent_y * cell_size + cell_size / 2)), int(cell_size / 3))

def draw_q_table_panel(screen, x_offset, env, Q, font):
    """ Orta bölümde Q-Tablosunun genel görünümünü çizer. """
    cell_size = 40
    for y in range(env.h):
        for x in range(env.w):
            rect = pygame.Rect(x_offset + x * cell_size, y * cell_size, cell_size, cell_size)
            # Ajanın olduğu hücreyi vurgula
            if (x, y) == env.state:
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect)
                pygame.draw.rect(screen, (0,0,0), rect, 2)
            else:
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)
            # Her hücrenin en iyi eylemini ok ile göster
            best_a = np.argmax(Q[y, x, :])
            action_arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
            arrow = action_arrows.get(best_a, "")
            text_surf = font.render(arrow, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

def draw_info_panel(screen, width, height, s, Q, action, paused, speed, font, small_font):
    """ Sağ tarafta detaylı bilgi panelini çizer. """
    panel_x = width
    panel_width = 300
    pygame.draw.rect(screen, INFO_PANEL_COLOR, (panel_x, 0, panel_width, height))
    pygame.draw.line(screen, GRID_COLOR, (panel_x, 0), (panel_x, height), 2)

    y_pos = 20
    def draw_text(text, f=font, offset=0, color=TEXT_COLOR):
        nonlocal y_pos
        surf = f.render(text, True, color)
        screen.blit(surf, (panel_x + 15 + offset, y_pos))
        y_pos += surf.get_height() + 5

    draw_text("--- BİLGİ PANELİ ---")
    y_pos += 10
    draw_text(f"Ajan Konumu (x,y): {s}", f=small_font)
    y_pos += 20
    draw_text(f"--- Mevcut Konumdaki ---")
    draw_text(f"--- Q-Değerleri ---")
    action_labels = ["↑ (Yukarı)", "↓ (Aşağı)", "← (Sol)", "→ (Sağ)"]
    current_q_values = Q[s[1], s[0], :]
    for i, q_val in enumerate(current_q_values):
        is_chosen = (i == action)
        draw_text(f"{action_labels[i]}: {q_val:.3f}", f=small_font, offset=10, color=CHOSEN_ACTION_COLOR if is_chosen else TEXT_COLOR)

    y_pos += 20
    draw_text("--- KONTROLLER ---")
    status_text = "DURAKLATILDI" if paused else "OYNATILIYOR"
    status_color = (255, 200, 0) if paused else (0, 255, 150)
    draw_text(f"Durum: {status_text}", f=small_font, color=status_color)
    draw_text(f"Hız: {speed} FPS", f=small_font)
    y_pos += 10
    draw_text("[SPACE] : Oynat/Duraklat", f=small_font)
    draw_text("[→]     : Sonraki Adım", f=small_font)
    draw_text("[↑] / [↓] : Hızı Değiştir", f=small_font)

def main():
    # 1. Adım: Ajanı eğit ve Q-Tablosunu al
    env, Q_table, _ = train()
    s = env.reset()

    # 2. Adım: Pygame'i başlat
    pygame.init()
    screen_width, screen_height = 1100, 500
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Q-Learning Görselleştirme ve Analiz Aracı")
    font = pygame.font.SysFont("monospace", 16, bold=True)
    small_font = pygame.font.SysFont("monospace", 14)

    grid_panel_w = 500
    q_table_panel_w = 250
    cell_size = (grid_panel_w - 100) / env.w

    # 3. Adım: Ana döngüyü başlat
    running = True
    clock = pygame.time.Clock()
    paused = True
    speed = 2
    needs_step = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_RIGHT and paused: needs_step = True
                if event.key == pygame.K_UP: speed = min(10, speed + 1)
                if event.key == pygame.K_DOWN: speed = max(1, speed - 1)

        if needs_step or not paused:
            x, y = s
            action = int(np.argmax(Q_table[y, x, :]))
            s_next, _, done, _ = env.step(action)
            s = s_next
            if done:
                s = env.reset()
                pygame.time.wait(500)
            needs_step = False

        # Çizim
        screen.fill(BG_COLOR)
        draw_grid_panel(screen, env, cell_size)
        draw_q_table_panel(screen, grid_panel_w, env, Q_table, font)
        draw_info_panel(screen, grid_panel_w + q_table_panel_w, screen_height, s, Q_table, action, paused, speed, font, small_font)
        pygame.display.flip()
        clock.tick(speed)

    pygame.quit()

if __name__ == "__main__":
    main()
