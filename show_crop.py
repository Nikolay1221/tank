"""Показывает оригинальный экран и обрезанное игровое поле."""
import cv2
import numpy as np
from python.battle_city_env import BattleCityEnv

env = BattleCityEnv(render_mode='rgb_array', start_level=1)
env.reset()

# Прогоняем 300 кадров чтобы враги появились
for _ in range(300):
    env.step(0)

screen = env.env.screen.copy()  # (240, 256, 3)
cropped = screen[16:224, 16:224]  # (208, 208, 3)

# Рисуем рамку обрезки на оригинале
original = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
cv2.rectangle(original, (16, 8), (224, 216), (0, 255, 0), 2)
cv2.putText(original, "CROP AREA", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Сохраняем
cv2.imwrite("crop_original.png", cv2.resize(original, (512, 480), interpolation=cv2.INTER_NEAREST))
cv2.imwrite("crop_result.png", cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), (416, 416), interpolation=cv2.INTER_NEAREST))

print(f"Original: {screen.shape}")
print(f"Cropped:  {cropped.shape}")
print("Saved: crop_original.png, crop_result.png")
env.close()
