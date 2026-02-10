"""
Battle City - Ручная игра с визуализацией A* пути.
WASD/Стрелки = Движение, Space = Огонь, Esc = Выход
"""

import pygame
import numpy as np
from python.battle_city_env import BattleCityEnv
from python.config import RLConfig
from python.astar import (
    build_grid_with_tanks, world_to_grid, compute_astar_reward,
    get_priority_enemy_grid,
    FIELD_OFFSET_X, FIELD_OFFSET_Y, BLOCK_PX, GRID_SIZE
)

FPS = 60
SCALE = 2  # Масштаб отображения (256*2=512, 240*2=480)

def draw_grid_overlay(surface, grid, path, player_grid, enemy_grid):
    """Рисует A* путь и grid на поверхности pygame."""
    # Рисуем путь (зелёные точки)
    if path:
        for i, (gx, gy) in enumerate(path):
            px = (FIELD_OFFSET_X + gx * BLOCK_PX + BLOCK_PX // 2) * SCALE
            py = (FIELD_OFFSET_Y + gy * BLOCK_PX + BLOCK_PX // 2) * SCALE

            if i == 0:
                # Старт — жёлтый
                color = (255, 255, 0)
                radius = 5
            elif i == len(path) - 1:
                # Цель — красный
                color = (255, 0, 0)
                radius = 6
            else:
                # Путь — зелёный
                color = (0, 255, 0)
                radius = 3

            pygame.draw.circle(surface, color, (px, py), radius)

        # Линии пути
        if len(path) >= 2:
            points = [
                ((FIELD_OFFSET_X + gx * BLOCK_PX + BLOCK_PX // 2) * SCALE,
                 (FIELD_OFFSET_Y + gy * BLOCK_PX + BLOCK_PX // 2) * SCALE)
                for gx, gy in path
            ]
            pygame.draw.lines(surface, (0, 200, 0), False, points, 2)

    # Стены (красные полупрозрачные квадраты) — рисуем тонкую рамку
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            if not grid[gy, gx]:
                px = (FIELD_OFFSET_X + gx * BLOCK_PX) * SCALE
                py = (FIELD_OFFSET_Y + gy * BLOCK_PX) * SCALE
                rect = pygame.Rect(px, py, BLOCK_PX * SCALE, BLOCK_PX * SCALE)
                pygame.draw.rect(surface, (255, 0, 0), rect, 1)


def main():
    pygame.init()
    screen = pygame.display.set_mode((512, 480))
    pygame.display.set_caption("Battle City - A* Path Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    env = BattleCityEnv(render_mode='rgb_array', start_level=1)
    obs, info = env.reset()

    total_reward = 0.0
    total_astar_reward = 0.0
    episode = 1
    running = True
    prev_player_grid = None

    print(f"\n--- Episode {episode} ---")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break

        # Действие
        fire = keys[pygame.K_SPACE] or keys[pygame.K_j]
        up = keys[pygame.K_w] or keys[pygame.K_UP]
        down = keys[pygame.K_s] or keys[pygame.K_DOWN]
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        if fire and up: action = 6
        elif fire and right: action = 7
        elif fire and down: action = 8
        elif fire and left: action = 9
        elif fire: action = 5
        elif up: action = 1
        elif right: action = 2
        elif down: action = 3
        elif left: action = 4
        else: action = 0

        if keys[pygame.K_p]:
            print(f"\n--- DEBUG P ---")
            print(f"Path: {path}")
            print(f"Target enemy: {enemy_grid}")
            if path and len(path) > 1:
                chk_node = path[1]
                val = grid[chk_node[1], chk_node[0]]
                print(f"Node {chk_node}: grid={val}")
            
            # Print problematic row 1
            row_str = ""
            for x in range(13):
                row_str += "#" if not grid[1, x] else "."
            print(f"Row 1 grid: {row_str}")
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # A* Reward
        nes_screen = env.env.screen.copy()
        astar_reward, path, player_grid = compute_astar_reward(
            nes_screen, env.env.ram, prev_player_grid
        )
        total_astar_reward += astar_reward
        prev_player_grid = player_grid

        # Grid для отрисовки стен
        grid = build_grid_with_tanks(nes_screen, env.env.ram)

        # Ближайший враг для визуализации (теперь приоритетный)
        enemy_grid = get_priority_enemy_grid(env.env.ram, grid)

        if astar_reward != 0:
            print(f"  A*: {astar_reward:+.2f}  total_a*: {total_astar_reward:.1f}  "
                  f"kills={info.get('kills', 0)}  path_len={len(path)}")

        if done:
            print(f"  === EP {episode} END === Kills:{env.cumulative_kills}  "
                  f"Reward:{total_reward:.0f}  A*:{total_astar_reward:.1f}")
            episode += 1
            obs, info = env.reset()
            total_reward = 0.0
            total_astar_reward = 0.0
            prev_player_grid = None
            print(f"\n--- Episode {episode} ---")
            continue

        # Рендер
        frame = env.env.render(mode='rgb_array')
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (512, 480))
        screen.blit(surf, (0, 0))

        # A* overlay
        draw_grid_overlay(screen, grid, path, player_grid, enemy_grid)

        # Текстовый overlay
        ram_kills = [int(env.env.ram[addr]) for addr in RLConfig.ADDR_KILLS]
        lines = [
            f"Kills:{env.cumulative_kills}  Lives:{env.env.ram[RLConfig.ADDR_LIVES]}  "
            f"Stage:{env.env.ram[RLConfig.ADDR_STAGE]}  Left:{env.env.ram[0x80]}",
            f"A*Reward:{total_astar_reward:.1f}  Path:{len(path)}  "
            f"GameReward:{total_reward:.0f}  Ep:{episode}",
        ]
        for i, line in enumerate(lines):
            text_surf = font.render(line, True, (0, 255, 0), (0, 0, 0))
            screen.blit(text_surf, (4, 4 + i * 16))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()
