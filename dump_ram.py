"""Точная визуализация тайлового буфера.
Пробуем разные начальные адреса и stride."""
import numpy as np
from python.battle_city_env import BattleCityEnv

env = BattleCityEnv(render_mode='rgb_array', start_level=1)
env.reset()
for _ in range(300):
    env.step(0)

ram = env.env.ram

# Значения: $00=пусто, $0F=кирпич, $10=сталь, $11=??, $20=вода, $6A=база
# %=$11, #=$0F, ==$10, ~~=$20, E=$6A

# Ранее увидели что $0400 начинается с $11 (26 штук).
# $11 может быть бордюр. Попробуем начать после бордюра.

# Попробуем $0400 + 26*2 (пропуск 2 строк бордюра) + 2 (пропуск 2 бордюрных столбцов)
# = $0400 + 54 = $0436

# Level 1 Battle City map (для сравнения):
# Верхняя часть в основном пустая, затем кирпичные колонны
print("=== Testing: MAP at $0400, row_stride=26, field=26x26 ===")
print("    Legend: .=00(empty) #=0F(brick) ==10(steel) ~=20(water) B=6A(base) @=11(?) %=other")
# Trying to display as 26x26 with proper border skipping
for row in range(26):
    vis = ""
    for col in range(26):
        addr = 0x0400 + row * 26 + col
        val = ram[addr]
        if val == 0x00: vis += "."
        elif val == 0x0F: vis += "#"
        elif val == 0x10: vis += "="
        elif val == 0x11: vis += "@"
        elif val == 0x13: vis += "T"  # trees?
        elif val == 0x14: vis += "T"
        elif val == 0x20: vis += "~"
        elif val == 0x58: vis += "?"
        elif val == 0x6A: vis += "B"
        else: vis += f"{val:X}"
    print(f"  {row:2d} |{vis}|")

# Also try $0500 with stride=32 (power of 2)
print("\n=== Testing: MAP at $0400, stride=32 ===")
for row in range(26):
    vis = ""
    for col in range(26):
        addr = 0x0400 + row * 32 + col
        if addr < len(ram):
            val = ram[addr]
        else:
            val = 0xFF
        if val == 0x00: vis += "."
        elif val == 0x0F: vis += "#"
        elif val == 0x10: vis += "="
        elif val == 0x11: vis += "@"
        elif val == 0x20: vis += "~"
        elif val == 0x6A: vis += "B"
        else: vis += f"{val:X}"
    print(f"  {row:2d} |{vis}|")

env.close()
