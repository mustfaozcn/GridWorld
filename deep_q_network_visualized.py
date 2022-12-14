
from __future__ import annotations
# -*- coding: utf-8 -*-
"""
DQN'in Karar Verme Sürecinin Pygame ile Görselleştirilmesi
---------------------------------------------------------
Bu dosya, eğitilmiş bir DQN ajanının her bir adımda sinir ağı içinde
neler olduğunu görsel olarak anlamak için tasarlanmıştır.

- Ajanın haritadaki konumu (durum) sinir ağına nasıl giriyor?
- Gizli katmanlardaki nöronlar nasıl aktive oluyor?
- Çıktı katmanı Q-değerlerini nasıl üretiyor?
- En iyi eylem nasıl seçiliyor?

Bu soruların cevaplarını animasyonlu bir şekilde gösterir.

Çalıştırmak için:
1. Gerekliyse `pip install pygame` ile kütüphaneyi yükleyin.
2. Bu dosyayı normal bir Python scripti olarak çalıştırın.
3. Önce modelin eğitilmesi için konsolu takip edin.
4. Eğitim bitince Pygame penceresi açılacaktır.
"""
import pygame
import numpy as np

# deep_q_network_numpy.py dosyasından import edilen sınıflar ve fonksiyonlar
from deep_q_network_numpy import (
    MLP,
    train_dqn
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
NEURON_COLOR = (100, 100, 200)
ACTIVE_NEURON_COLOR = (255, 100, 150)
CHOSEN_ACTION_COLOR = (0, 255, 150)
LOW_ACTIVATION_COLOR = (50, 50, 90)
HIGH_ACTIVATION_COLOR = (250, 250, 255)

def draw_grid(screen, env, cell_size):
    """ GridWorld haritasını çizer. """
    for y in range(env.h):
        for x in range(env.w):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

    # Hedefi çiz
    goal_x, goal_y = env.goal
    goal_rect = pygame.Rect(goal_x * cell_size + 5, goal_y * cell_size + 5, cell_size - 10, cell_size - 10)
    pygame.draw.rect(screen, GOAL_COLOR, goal_rect)

    # Ajanı çiz
    agent_x, agent_y = env.state
    pygame.draw.circle(screen, AGENT_COLOR, (int(agent_x * cell_size + cell_size / 2), int(agent_y * cell_size + cell_size / 2)), int(cell_size / 3))

def draw_network(screen, x_offset, y_offset, activations, chosen_action, font):
    """ Sinir ağını ve aktivasyonları çizer. """
    x, z1, a1, z2, a2, q_values = activations
    q_values = q_values[0] # Batch boyutunu kaldır
    a1 = a1[0]; a2 = a2[0] # Batch boyutunu kaldır

    layer_dims = [2, 10, 10, 4] # Görselleştirme için nöron sayısı (tamamı değil)
    layer_spacing = 150
    neuron_radius = 12

    layer_neurons = []
    # Nöron pozisyonlarını hesapla
    for i, dim in enumerate(layer_dims):
        neurons = []
        layer_height = dim * (neuron_radius * 2 + 10)
        start_y = y_offset + (400 - layer_height) / 2
        for j in range(dim):
            pos = (x_offset + i * layer_spacing, start_y + j * (neuron_radius * 2 + 10))
            neurons.append(pos)
        layer_neurons.append(neurons)

    # Bağlantıları çiz (arka planda)
    for i in range(len(layer_neurons) - 1):
        for n1_pos in layer_neurons[i]:
            for n2_pos in layer_neurons[i+1]:
                pygame.draw.line(screen, GRID_COLOR, n1_pos, n2_pos, 1)

    # Nöronları ve aktivasyonları çiz
    action_labels = ["↑", "↓", "←", "→"]
    for i, layer_pos in enumerate(layer_neurons):
        for j, pos in enumerate(layer_pos):
            val = 0
            is_active = False
            # Aktivasyon değerlerini al ve renklendir
            if i == 0: # Girdi
                val = x[0][j]
                color = tuple(int(c * val) for c in HIGH_ACTIVATION_COLOR)
            elif i == 1: # Gizli Katman 1 (a1'den örneklem)
                val = a1[j * (len(a1)//len(layer_pos))] # a1'den temsili nöron al
            elif i == 2: # Gizli Katman 2 (a2'den örneklem)
                val = a2[j * (len(a2)//len(layer_pos))]
            elif i == 3: # Çıktı
                val = q_values[j]
                is_active = (j == chosen_action)

            # ReLU sonrası aktivasyonlar için renk (0'dan büyükse)
            if i > 0 and i < 3:
                norm_val = min(val / (a1.mean() + 1e-6), 1.0) if val > 0 else 0
                color = (
                    LOW_ACTIVATION_COLOR[0] + (HIGH_ACTIVATION_COLOR[0] - LOW_ACTIVATION_COLOR[0]) * norm_val,
                    LOW_ACTIVATION_COLOR[1] + (HIGH_ACTIVATION_COLOR[1] - LOW_ACTIVATION_COLOR[1]) * norm_val,
                    LOW_ACTIVATION_COLOR[2] + (HIGH_ACTIVATION_COLOR[2] - LOW_ACTIVATION_COLOR[2]) * norm_val,
                )
            elif i == 0: # Girdi için renk
                color = (50 + 200 * val, 50 + 200 * val, 50 + 200 * val)


            pygame.draw.circle(screen, color if i > 0 else NEURON_COLOR, pos, neuron_radius)
            pygame.draw.circle(screen, CHOSEN_ACTION_COLOR if is_active else TEXT_COLOR, pos, neuron_radius, 2 if is_active else 1)

            # Çıktı nöronlarına Q-değerlerini ve Eylem etiketlerini yaz
            if i == 3:
                q_text = f"{val:.2f}"
                text_surf = font.render(q_text, True, CHOSEN_ACTION_COLOR if is_active else TEXT_COLOR)
                screen.blit(text_surf, (pos[0] + 20, pos[1] - 8))
                action_surf = font.render(action_labels[j], True, TEXT_COLOR)
                screen.blit(action_surf, (pos[0] - 40, pos[1] - 8))


def main():
    # 1. Adım: Ajanı eğit
    env, policy_net, _, _ = train_dqn()
    s = env.reset()

    # 2. Adım: Pygame'i başlat
    pygame.init()
    screen_width, screen_height = 1200, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("DQN Görselleştirme")
    font = pygame.font.SysFont("monospace", 16, bold=True)
    small_font = pygame.font.SysFont("monospace", 14)

    cell_size = (screen_height - 100) / env.h
    grid_width = cell_size * env.w
    net_x_offset = grid_width + 100

    # 3. Adım: Ana döngüyü başlat
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Ajanın karar verme süreci
        s_vec = to_state_vector(s, env.w, env.h)
        # İleri yayılımı yap ve ara katman değerlerini al
        q_values, cache = policy_net.forward(s_vec[None, :])
        x, z1, a1, z2, a2 = cache
        action = int(np.argmax(q_values))

        # Çizim işlemleri
        screen.fill(BG_COLOR)
        draw_grid(screen, env, cell_size)
        draw_network(screen, net_x_offset, 50, (x, z1, a1, z2, a2, q_values), action, font)

        # Bilgi metinlerini yazdır
        state_text = f"Durum (x,y): {s}"
        norm_text = f"Normalize Vektör: [{s_vec[0]:.2f}, {s_vec[1]:.2f}]"
        text1 = font.render(state_text, True, TEXT_COLOR)
        text2 = small_font.render(norm_text, True, TEXT_COLOR)
        screen.blit(text1, (10, screen_height - 60))
        screen.blit(text2, (10, screen_height - 35))


        pygame.display.flip()

        # Bir sonraki adıma geç
        s, _, done, _ = env.step(action)
        if done:
            s = env.reset()
            pygame.time.wait(1000) # Hedefe ulaşınca bekle

        # Animasyon hızını ayarla
        clock.tick(2) # Saniyede 2 adım

    pygame.quit()

if __name__ == "__main__":
    main()
