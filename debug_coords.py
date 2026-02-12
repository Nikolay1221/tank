"""Эмпирический поиск правильного маппинга координат RAM → тайлы.
Берём координаты живых танков и проверяем все возможные strides и offsets."""
import numpy as np
from python.battle_city_env import BattleCityEnv
from python.config import RLConfig

env = BattleCityEnv(render_mode='rgb_array', start_level=1)
env.reset()
for _ in range(300):
    env.step(0)

ram = env.env.ram

# Считываем координаты всех живых танков
tanks = []
for i in range(8):
    status = ram[RLConfig.ADDR_ENEMY_STATUS_BASE + i]
    x = int(ram[RLConfig.ADDR_COORD_X_BASE + i])
    y = int(ram[RLConfig.ADDR_COORD_Y_BASE + i])
    if status >= 128 and (x > 0 or y > 0):
        label = "P1" if i == 0 else f"E{i-2}" if i >= 2 else f"P{i+1}"
        cx, cy = x + 8, y + 8
        tanks.append((label, x, y, cx, cy))
        print(f"{label}: RAM({x},{y}) center({cx},{cy})")

print(f"\nTanks found: {len(tanks)}")

# Все танки должны быть на проходимых клетках
# Попробуем разные комбинации offset и stride
PASSABLE = {0x00, 0x11, 0x13, 0x14, 0x58}

print("\n=== Brute force: trying all offsets and strides ===")
best_results = []

for stride in [26, 28, 30, 32]:
    for px_offset in range(0, 24, 2):
        for py_offset in range(0, 24, 2):
            all_passable = True
            for label, rx, ry, cx, cy in tanks:
                # pixel → half-tile
                htx = (cx - px_offset) // 8
                hty = (cy - py_offset) // 8
                
                if htx < 0 or hty < 0 or htx >= 32 or hty >= 32:
                    all_passable = False
                    break
                
                addr = 0x0400 + hty * stride + htx
                if addr >= len(ram):
                    all_passable = False
                    break
                
                tile = ram[addr]
                if tile not in PASSABLE:
                    all_passable = False
                    break
            
            if all_passable:
                best_results.append((px_offset, py_offset, stride))

print(f"\nMatching combos: {len(best_results)}")
for px_off, py_off, stride in best_results[:20]:
    print(f"  pixel_offset=({px_off},{py_off}), stride={stride}")
    for label, rx, ry, cx, cy in tanks:
        htx = (cx - px_off) // 8
        hty = (cy - py_off) // 8
        addr = 0x0400 + hty * stride + htx
        tile = ram[addr]
        print(f"    {label}: center({cx},{cy}) → ht({htx},{hty}) tile=${tile:02X}")

env.close()
