
from __future__ import annotations
# -*- coding: utf-8 -*-
"""Transformer Görselleştirme"""
import pygame
import numpy as np
import random
from transformer_numpy import to_feature_sequence, Transformer, ReplayBuffer, train_transformer_dqn
from common import GridWorld, epsilon_greedy

BG_COLOR = (10, 10, 40)
GRID_COLOR = (80, 80, 120)
AGENT_COLOR = (255, 200, 0)
GOAL_COLOR = (0, 255, 150)
TEXT_COLOR = (240, 240, 240)

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

def draw_attention_heatmap(screen, attn_weights, x_offset, y_offset, title, font):
    if attn_weights is None or len(attn_weights) == 0:
        return
    small_font = pygame.font.SysFont("monospace", 12)
    title_surf = small_font.render(title, True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset - 20))
    
    # İlk head'in attention weight'lerini göster
    if len(attn_weights) > 0:
        attn = attn_weights[0]
        num_tokens = attn.shape[0]
        cell_size = 40
        for i in range(num_tokens):
            for j in range(num_tokens):
                weight = attn[i, j]
                intensity = int(255 * weight)
                color = (intensity, intensity // 2, 0)
                rect = pygame.Rect(x_offset + j * cell_size, y_offset + i * cell_size, cell_size - 2, cell_size - 2)
                pygame.draw.rect(screen, color, rect)

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
    env, policy_net, _, _ = train_transformer_dqn()
    s = env.reset()
    
    pygame.init()
    screen_width, screen_height = 1400, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Transformer-DQN Görselleştirme")
    font = pygame.font.SysFont("monospace", 16, bold=True)
    
    cell_size = 60
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        seq = to_feature_sequence(s, env.w, env.h, env.goal)
        q_values, attn_weights = policy_net.forward(seq)
        action = int(np.argmax(q_values))
        
        screen.fill(BG_COLOR)
        draw_grid(screen, env, cell_size)
        draw_attention_heatmap(screen, attn_weights, 400, 50, "Self-Attention Weights", font)
        draw_q_values(screen, q_values, action, 10, 350, font)
        
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

