
from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Attention'in Karar Verme Sürecinin Pygame ile Görselleştirilmesi
"""
import pygame
import numpy as np
import random

# attention_numpy.py'den temel sınıflar
from attention_numpy import to_feature_vector, AttentionNet, ReplayBuffer, train_attention_dqn
from common import GridWorld, epsilon_greedy

# Renkler
BG_COLOR = (10, 10, 40)
GRID_COLOR = (80, 80, 120)
AGENT_COLOR = (255, 200, 0)
GOAL_COLOR = (0, 255, 150)
TEXT_COLOR = (240, 240, 240)
ATTENTION_COLOR = (255, 150, 100)

def draw_grid(screen, env, cell_size):
    for y in range(env.h):
        for x in range(env.w):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
    goal_x, goal_y = env.goal
    goal_rect = pygame.Rect(goal_x * cell_size + 5, goal_y * cell_size + 5, cell_size - 10, cell_size - 10)
    pygame.draw.rect(screen, GOAL_COLOR, goal_rect)
    agent_x, agent_y = env.state
    pygame.draw.circle(screen, AGENT_COLOR, (int(agent_x * cell_size + cell_size / 2), int(agent_y * cell_size + cell_size / 2)), int(cell_size / 3))

def draw_attention_heatmap(screen, attn_weights, feature_names, x_offset, y_offset, title, font):
    """ Attention weight'lerini heatmap olarak çizer. """
    if attn_weights is None or len(attn_weights) == 0:
        return
    
    small_font = pygame.font.SysFont("monospace", 12)
    title_surf = small_font.render(title, True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset - 20))
    
    # İlk head'in attention weight'lerini göster
    attn = attn_weights[0]  # (num_features, num_features)
    num_features = attn.shape[0]
    
    cell_size = 30
    for i in range(num_features):
        for j in range(num_features):
            weight = attn[i, j]
            # Ağırlığı 0-255 arasına ölçeklendir
            intensity = int(255 * weight)
            color = (intensity, intensity // 2, 0)
            rect = pygame.Rect(x_offset + j * cell_size, y_offset + i * cell_size, cell_size - 2, cell_size - 2)
            pygame.draw.rect(screen, color, rect)
            
            # Değeri yaz
            if weight > 0.1:
                text = f"{weight:.2f}"
                text_surf = small_font.render(text, True, TEXT_COLOR)
                screen.blit(text_surf, (x_offset + j * cell_size + 5, y_offset + i * cell_size + 5))

def draw_features(screen, features, feature_names, x_offset, y_offset, title, font):
    """ Özellik vektörünü çizer. """
    small_font = pygame.font.SysFont("monospace", 12)
    title_surf = small_font.render(title, True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset - 20))
    
    for i, (name, val) in enumerate(zip(feature_names, features)):
        y_pos = y_offset + i * 25
        text = f"{name}: {val:.3f}"
        text_surf = small_font.render(text, True, TEXT_COLOR)
        screen.blit(text_surf, (x_offset, y_pos))
        
        # Bar chart
        bar_width = int(abs(val) * 150)
        color = (100, 200, 255) if val > 0 else (255, 100, 100)
        pygame.draw.rect(screen, color, (x_offset + 120, y_pos + 2, bar_width, 15))

def draw_q_values(screen, q_values, chosen_action, x_offset, y_offset, font):
    action_labels = ["↑", "↓", "←", "→"]
    action_names = ["Yukarı", "Aşağı", "Sol", "Sağ"]
    small_font = pygame.font.SysFont("monospace", 14)
    title_surf = small_font.render("Q-Değerleri:", True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset))
    
    for i, (label, name, q_val) in enumerate(zip(action_labels, action_names, q_values)):
        y_pos = y_offset + 25 + i * 25
        color = (0, 255, 150) if i == chosen_action else TEXT_COLOR
        text = f"{label} {name}: {q_val:.2f}"
        text_surf = small_font.render(text, True, color)
        screen.blit(text_surf, (x_offset, y_pos))

def main():
    env, policy_net, _, _ = train_attention_dqn()
    s = env.reset()
    
    pygame.init()
    screen_width, screen_height = 1400, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Attention-DQN Görselleştirme")
    font = pygame.font.SysFont("monospace", 16, bold=True)
    small_font = pygame.font.SysFont("monospace", 14)
    
    cell_size = 60
    feature_names = ["Agent_X", "Agent_Y", "Goal_X", "Goal_Y", "Distance", "Wall_X", "Wall_Y"]
    
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        features = to_feature_vector(s, env.w, env.h, env.goal)
        q_values, cache = policy_net.forward(features[None, :])
        q_values = q_values[0]
        attn_weights = cache['attention_weights'][0]  # (num_heads, num_features, num_features)
        action = int(np.argmax(q_values))
        
        screen.fill(BG_COLOR)
        draw_grid(screen, env, cell_size)
        draw_features(screen, features, feature_names, 400, 50, "Özellik Vektörü", font)
        draw_attention_heatmap(screen, attn_weights, feature_names, 700, 50, "Attention Weights (Head 0)", font)
        draw_q_values(screen, q_values, action, 10, 400, font)
        
        state_text = f"Durum (x,y): {s}"
        text1 = font.render(state_text, True, TEXT_COLOR)
        screen.blit(text1, (10, screen_height - 60))
        
        action_text = f"Seçilen Eylem: {['↑ Yukarı', '↓ Aşağı', '← Sol', '→ Sağ'][action]}"
        text2 = font.render(action_text, True, (0, 255, 150))
        screen.blit(text2, (10, screen_height - 35))
        
        pygame.display.flip()
        
        s, _, done, _ = env.step(action)
        if done:
            s = env.reset()
            pygame.time.wait(1000)
        
        clock.tick(2)
    
    pygame.quit()

if __name__ == "__main__":
    main()

