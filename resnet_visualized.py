
from __future__ import annotations
# -*- coding: utf-8 -*-
"""ResNet Görselleştirme"""
import pygame
import numpy as np
import random
from resnet_numpy import ResNet, ReplayBuffer, train_resnet_dqn
from common import GridWorld, to_state_vector, epsilon_greedy

BG_COLOR = (10, 10, 40)
GRID_COLOR = (80, 80, 120)
AGENT_COLOR = (255, 200, 0)
GOAL_COLOR = (0, 255, 150)
TEXT_COLOR = (240, 240, 240)
SKIP_COLOR = (255, 150, 100)

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

def draw_network_structure(screen, net, x_offset, y_offset, font):
    """ ResNet yapısını çizer - skip connection'ları gösterir """
    small_font = pygame.font.SysFont("monospace", 12)
    title_surf = small_font.render("ResNet Yapısı (Skip Connections)", True, TEXT_COLOR)
    screen.blit(title_surf, (x_offset, y_offset))
    
    # Katmanları çiz
    layer_spacing = 60
    for i in range(net.num_blocks + 2):
        y_pos = y_offset + 40 + i * layer_spacing
        layer_name = "Input" if i == 0 else f"Block {i}" if i <= net.num_blocks else "Output"
        text_surf = small_font.render(layer_name, True, TEXT_COLOR)
        screen.blit(text_surf, (x_offset, y_pos))
        
        # Skip connection çiz
        if i > 0 and i <= net.num_blocks:
            # Skip connection çizgisi
            pygame.draw.line(screen, SKIP_COLOR, (x_offset + 100, y_pos - layer_spacing), 
                           (x_offset + 100, y_pos), 3)

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
    env, policy_net, _, _ = train_resnet_dqn()
    s = env.reset()
    
    pygame.init()
    screen_width, screen_height = 1200, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("ResNet-DQN Görselleştirme")
    font = pygame.font.SysFont("monospace", 16, bold=True)
    
    cell_size = 60
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        s_vec = to_state_vector(s, env.w, env.h)
        q_values, _ = policy_net.forward(s_vec[None, :])
        q_values = q_values[0]
        action = int(np.argmax(q_values))
        
        screen.fill(BG_COLOR)
        draw_grid(screen, env, cell_size)
        draw_network_structure(screen, policy_net, 400, 50, font)
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

