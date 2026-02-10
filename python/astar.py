"""
A* Pathfinding для Battle City.
Строит карту проходимости из экранных пикселей и ищет путь к ближайшему врагу.
"""

import numpy as np
import heapq
from .config import RLConfig

# Размер игрового поля в блоках (13x13)
GRID_SIZE = 13
BLOCK_PX = 16  # Пикселей на блок
FIELD_OFFSET_X = 16  # Начало поля в пикселях (от левого края экрана)
FIELD_OFFSET_Y = 24  # Смещение по вертикали (визуально ~1.5 блока от верха) карты в RAM
# Адрес тайлового буфера карты в RAM (не используется для grid, только координаты)
MAP_RAM_BASE = 0x0400
MAP_RAM_STRIDE = 32
MAP_BORDER = 0

# Точки сэмплирования внутри блока (смещения от верхнего левого угла)
# Добавляем центр (8, 8) для надежности
_SAMPLE_OFFSETS = [(4, 4), (4, 12), (12, 4), (12, 12), (8, 8)]

# Цвета (примерные пороги)
def _is_passable_pixel(r, g, b):
    """Определяет, является ли пиксель проходимым (ЧЕРНЫЙ ПУТЬ)."""
    # Cast to int to prevent overflow (uint8)
    r, g, b = int(r), int(g), int(b)
    val = r + g + b
    
    # Черный (путь) - проходимо
    if val < 80: return True
    
    # Зеленый (лес) - проходимо (иначе танки застрянут в лесу)
    # G > R+20, G > B+20. Лес в Battle City: R=0,G=158,B=0? Или темнее.
    if g > r + 20 and g > b + 20: return True
    
    # Лед (белый/серый)? Если пользователь сказал "только путь черный", то лед - стена?
    # Танки скользят по льду. Но лучше пока считать проходимым.
    # Лед ~ (180, 180, 200). > 500 val.
    # Пусть пока только черный и зеленый.
    
    return False


# Сдвиг координат спрайтов (визуально танки левее/правее RAM?)
# Наблюдение: точки маппинга (gx) правее танков. Нужно уменьшить gx -> уменьшить x.
SPRITE_X_CORRECTION = -6


def get_tank_grid(ram, x_addr, y_addr):
    """Читает координаты танка из RAM, применяет коррекцию и возвращает grid (gx, gy)."""
    tx = int(ram[x_addr]) + SPRITE_X_CORRECTION
    ty = int(ram[y_addr])
    return world_to_grid(tx, ty)


def build_grid_from_colors(screen, ram):
    """Строит grid проходимости: ЧЕРНЫЙ ЦВЕТ = ПУТЬ. Остальное - стена (кроме танков)."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    h, w = screen.shape[:2]

    # Создаем маску танков
    tank_cells = set()
    for i in range(8):
        status = ram[RLConfig.ADDR_ENEMY_STATUS_BASE + i]
        if status < 128: continue
        
        # Используем helper с коррекцией
        # Для врагов адреса идут массивом, но они последовательны?
        # В RLConfig адреса - это базы.
        # X: 0x0090, 0x0091...
        # Y: 0x0098, 0x0099...
        
        pos = get_tank_grid(ram, 
                            RLConfig.ADDR_COORD_X_BASE + i, 
                            RLConfig.ADDR_COORD_Y_BASE + i)
        
        if pos is not None:
            gx, gy = pos
            # Углы танка для маски
            # Тут нужна более сложная логика, если мы хотим маскировать углы
            # Но get_tank_grid возвращает одну точку (верх-лево с коррекцией).
            # Давайте просто замаскируем 2x2 вокруг этой точки, так надежнее для цвета.
            # Или лучше пересчитать углы с коррекцией?
            
            # Пересчитаем углы с коррекцией
            base_tx = int(ram[RLConfig.ADDR_COORD_X_BASE + i]) + SPRITE_X_CORRECTION
            base_ty = int(ram[RLConfig.ADDR_COORD_Y_BASE + i])
            
            corners = [
                (base_tx, base_ty),
                (base_tx + 15, base_ty),
                (base_tx, base_ty + 15),
                (base_tx + 15, base_ty + 15)
            ]
            
            for cx, cy in corners:
                p = world_to_grid(cx, cy)
                if p: tank_cells.add(p)

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            # Если клетка под танком - считаем проходимой
            if (gx, gy) in tank_cells:
                grid[gy, gx] = True
                continue

            bx = FIELD_OFFSET_X + gx * BLOCK_PX
            by = FIELD_OFFSET_Y + gy * BLOCK_PX

            passable_pixels = 0
            valid_samples = 0
            
            for ox, oy in _SAMPLE_OFFSETS:
                px, py = bx + ox, by + oy
                if py >= h or px >= w: continue
                
                valid_samples += 1
                r, g, b = screen[py, px]
                if _is_passable_pixel(r, g, b):
                    passable_pixels += 1

            # Блок проходим, если большинство пикселей черные (или зеленые)
            # >= 4 из 5 (строгий критерий, чтобы стены не "протекали")
            grid[gy, gx] = (passable_pixels >= 4)

    return grid


def build_grid_with_tanks(screen, ram):
    # Теперь это просто обертка
    return build_grid_from_colors(screen, ram)


def world_to_grid(x, y):
    """Преобразует экранные координаты (world) в grid (0..12)."""
    gx = int((x - FIELD_OFFSET_X) // BLOCK_PX)
    gy = int((y - FIELD_OFFSET_Y) // BLOCK_PX)
    
    if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
        return gx, gy
    return None


def grid_to_world(gx, gy):
    """Конвертирует grid-координаты в пиксельные (центр блока).

    Returns:
        (px, py): пиксельные координаты центра блока
    """
    px = FIELD_OFFSET_X + gx * BLOCK_PX + BLOCK_PX // 2
    py = FIELD_OFFSET_Y + gy * BLOCK_PX + BLOCK_PX // 2
    return (px, py)


def find_path(grid, start, goal):
    """A* поиск пути на grid.

    Args:
        grid: (13, 13) bool — True = проходимо
        start: (gx, gy) начальная позиция
        goal: (gx, gy) конечная позиция

    Returns:
        path: список (gx, gy) от start до goal, или [] если пути нет
    """
    if start is None or goal is None:
        return []
    if start == goal:
        return [start]
    if not grid[start[1], start[0]] or not grid[goal[1], goal[0]]:
        # Стартовая или конечная позиция непроходима — пропускаем проверку для старта
        # (танк может стоять на своей клетке даже если grid показывает стену)
        pass

    # A* с Manhattan distance
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # 4 направления: вверх, вниз, влево, вправо
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Восстанавливаем путь
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy

            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            if not grid[ny, nx]:
                continue

            neighbor = (nx, ny)
            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Пути нет


def get_priority_enemy_grid(ram, grid):
    """Находит наиболее приоритетного врага (угроза базе + близость).
    
    Приоритет:
    1. Близость врага к базе (защита орла).
    2. Близость врага к игроку (перехват).
    
    Score = dist(enemy, base) + 0.5 * dist(player, enemy).
    Минимальный score = высший приоритет.
    """
    player_pos = get_tank_grid(ram, RLConfig.ADDR_PLAYER_X, RLConfig.ADDR_PLAYER_Y)
    if not player_pos:
        return None

    # Позиция базы (орла) - (6, 12)
    BASE_POS = (6, 12)

    # Помечаем позицию игрока как проходимую
    grid_copy = grid.copy()
    grid_copy[player_pos[1], player_pos[0]] = True

    best_score = float('inf')
    best_pos = None

    for i in range(2, 8):  # Слоты врагов 2-7
        status = ram[RLConfig.ADDR_ENEMY_STATUS_BASE + i]
        if status < 128:  # Не активен
            continue

        # Используем helper с коррекцией
        enemy_pos = get_tank_grid(ram, 
                                  RLConfig.ADDR_COORD_X_BASE + i, 
                                  RLConfig.ADDR_COORD_Y_BASE + i)
        
        if enemy_pos is None:
            continue

        # Помечаем позицию врага как проходимую (цель)
        grid_copy[enemy_pos[1], enemy_pos[0]] = True

        # Считаем метрики
        # 1. Расстояние от врага до базы (Manhattan, так как враг может ломать стены)
        dist_to_base = abs(enemy_pos[0] - BASE_POS[0]) + abs(enemy_pos[1] - BASE_POS[1])

        # 2. Расстояние от игрока до врага
        path_to_enemy = find_path(grid_copy, player_pos, enemy_pos)
        if path_to_enemy:
            dist_to_player = len(path_to_enemy)
        else:
            # Если пути нет, берем Manhattan с штрафом (но не бесконечность)
            dist_to_player = (abs(player_pos[0] - enemy_pos[0]) + 
                              abs(player_pos[1] - enemy_pos[1])) * 2

        # Итоговый счет (чем меньше, тем приоритетнее/опаснее)
        # Вес базы выше (1.0), вес игрока ниже (0.5)
        score = dist_to_base + 0.5 * dist_to_player

        # Восстанавливаем grid (для следующего врага)
        grid_copy[enemy_pos[1], enemy_pos[0]] = grid[enemy_pos[1], enemy_pos[0]]

        if score < best_score:
            best_score = score
            best_pos = enemy_pos

    return best_pos


def compute_astar_reward(screen, ram, prev_player_grid):
    """Вычисляет A* reward за один шаг.

    Args:
        screen: NES экран (240, 256, 3) RGB
        ram: массив RAM эмулятора
        prev_player_grid: (gx, gy) предыдущая grid-позиция игрока

    Returns:
        reward: float - 0.5 если движемся по A* пути, -0.25 если удаляемся, 0 если стоим
        path: список (gx, gy) для визуализации
        player_grid: текущая grid-позиция игрока
    """
    grid = build_grid_with_tanks(screen, ram)

    player_pos = get_tank_grid(ram, RLConfig.ADDR_PLAYER_X, RLConfig.ADDR_PLAYER_Y)

    if player_pos is None:
        return 0.0, [], prev_player_grid

    # Ищем врага с наивысшим приоритетом (угроза базе)
    enemy_pos = get_priority_enemy_grid(ram, grid)
    if enemy_pos is None:
        return 0.0, [], player_pos

    # A* путь
    path = find_path(grid, player_pos, enemy_pos)

    if not path or len(path) < 2:
        return 0.0, path, player_pos

    # Вычисляем reward
    reward = 0.0
    if prev_player_grid is not None and prev_player_grid != player_pos:
        # Игрок двигался
        next_cell = path[1]  # Следующая клетка на пути

        # Расстояние до цели
        prev_dist = abs(prev_player_grid[0] - enemy_pos[0]) + abs(prev_player_grid[1] - enemy_pos[1])
        curr_dist = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])

        if player_pos == next_cell:
            # Точно на пути A*
            reward = 0.5
        elif curr_dist < prev_dist:
            # Приближаемся к врагу (но не точно по A*)
            reward = 0.25
        elif curr_dist > prev_dist:
            # Удаляемся от врага
            reward = -0.25

    return reward, path, player_pos
