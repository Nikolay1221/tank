import gym
import cv2
import time
import numpy as np
import collections
import ctypes
from python.battle_city_env import BattleCityEnv
from python.config import RLConfig

# --- GLOBAL STATE ---
scroll_y = 0
MAX_SCROLL = 0
is_paused = False
current_stage_req = 1 # Start at Stage 1
force_reset = False

def mouse_callback(event, x, y, flags, param):
    global scroll_y, MAX_SCROLL, is_paused, current_stage_req, force_reset
    
    # Scroll
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0: scroll_y = max(0, scroll_y - 20)
        else: scroll_y = min(MAX_SCROLL, scroll_y + 20)
        
    if event == cv2.EVENT_LBUTTONDOWN:
        # NOTE: y coords must match the render logic below
        
        # 1. PAUSE BUTTON (Approx y=80-120)
        if 20 <= x <= 220 and 80 <= y <= 120:
            is_paused = not is_paused
            
        # 2. LEVEL CONTROLS (Approx y=230-270, adjusted for layout)
        # We need to be careful with exact Y positions as layout shifts.
        # Let's fix the layout in the render loop first, then match here.
        # Using fixed offsets based on header (40) + pause (60) + 20 margin = 120 start for stats
        
        # Let's put Level Controls at y=140 (between Pause and Global Stats)
        # Prev: x=20-70, y=140-180
        # Next: x=170-220, y=140-180
        
        btn_y_start = 140
        btn_y_end = 180
        
        if btn_y_start <= y <= btn_y_end:
            # Prev
            if 20 <= x <= 70:
                current_stage_req = max(1, current_stage_req - 1)
                force_reset = True
            # Next
            elif 170 <= x <= 220:
                current_stage_req = min(35, current_stage_req + 1)
                force_reset = True

def main():
    global force_reset
    print("Starting RAM Debug Manager v2.7...")
    print("Controls: WASD=Move, SPACE=Fire, R=Reset, Q=Quit")
    
    # Start at Stage 1
    env = BattleCityEnv(render_mode='rgb_array', start_level=current_stage_req) 
    env.reset()
    
    # --- CONFIG ---
    GAME_W, GAME_H = 512, 480
    DEBUG_W, DEBUG_H = 600, 800
    
    # --- COLORS (BGR) ---
    C_BG = (20, 20, 20)
    C_PANEL = (40, 40, 40)
    C_TEXT = (220, 220, 220)
    C_ACCENT = (255, 165, 0) # Orange
    C_GREEN = (100, 255, 100)
    C_RED = (100, 100, 255)
    C_YELLOW = (0, 255, 255)
    C_GRAY = (120, 120, 120)
    C_BLUE = (255, 200, 100)
    C_BTN_HOVER = (60, 60, 60)
    C_BTN = (180, 180, 180)

    # --- REWARD LOG ---
    reward_log = collections.deque(maxlen=50) # Keep more history
    total_reward = 0.0

    # Create Windows
    cv2.namedWindow("Battle City Game", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Debug Manager", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Debug Manager", mouse_callback)

    try:
        while True:
            start_time = time.time()
            
            # --- INPUT ---
            def is_down(vk): return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0
            cv2.waitKey(1) 
            
            VK_W, VK_A, VK_S, VK_D = 0x57, 0x41, 0x53, 0x44
            VK_SPACE = 0x20
            VK_R, VK_Q = 0x52, 0x51
            
            action = 0
            if is_down(VK_Q): break
            elif is_down(VK_R) or force_reset: 
                # Apply new level if requested
                env.start_level = current_stage_req
                env.reset()
                
                reward_log.clear()
                total_reward = 0
                force_reset = False
                time.sleep(0.2)
                continue
                
            if is_down(VK_SPACE): action = 5
            elif is_down(VK_W): action = 1
            elif is_down(VK_D): action = 2
            elif is_down(VK_S): action = 3
            elif is_down(VK_A): action = 4
            
            # --- STEP (1x Speed) ---
            if not is_paused:
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                
                # Log
                if reward != 0:
                    sym = "+" if reward > 0 else ""
                    msg = f"{sym}{reward:.1f}"
                    col = C_GREEN if reward > 0 else C_RED
                    if reward >= 1.0: msg += " (KILL)"
                    elif reward <= -10: msg += " (DEATH)"
                    if abs(reward) > 0.05: reward_log.append((msg, col))
                
                # Auto Reset
                if done:
                    reward_log.append(("EPISODE DONE", C_ACCENT))
                    env.reset()
            else:
                 pass

            # --- RENDER GAME ---
            frame = env.render()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
                if is_paused:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (GAME_W, GAME_H), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    cv2.putText(frame, "PAUSED", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, C_YELLOW, 3)
                    
                cv2.imshow("Battle City Game", frame)

            # --- RENDER DEBUG MANAGER ---
            VIRTUAL_H = 1200 
            canvas = np.zeros((VIRTUAL_H, DEBUG_W, 3), dtype=np.uint8)
            canvas[:] = C_BG
            
            y = 40
            x = 20
            
            # HEADER
            cv2.putText(canvas, "DEBUG MANAGER (RAM)", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_ACCENT, 2)
            y += 40
            
            # PAUSE BUTTON
            btn_col = C_GREEN if not is_paused else C_RED
            btn_text = "PAUSE (ON)" if is_paused else "PAUSE (OFF)"
            cv2.rectangle(canvas, (x, y), (x+200, y+40), C_PANEL, -1)
            cv2.rectangle(canvas, (x, y), (x+200, y+40), btn_col, 2)
            cv2.putText(canvas, btn_text, (x+20, y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, btn_col, 2)
            y += 60
            
            # LEVEL SELECTOR (Interjected here)
            # Layout: [ < ] STAGE: N [ > ]
            # Y range: 140 - 180
            
            # Prev Button [<]
            cv2.rectangle(canvas, (x, y), (x+50, y+40), C_PANEL, -1)
            cv2.rectangle(canvas, (x, y), (x+50, y+40), C_BTN, 1)
            cv2.putText(canvas, "<", (x+15, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_BTN, 2)
            
            # Text "STAGE: N"
            cv2.putText(canvas, f"STAGE: {current_stage_req}", (x+65, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_YELLOW, 2)
            
            # Next Button [>]
            # x_next = x + 150
            x_next = x + 170
            cv2.rectangle(canvas, (x_next, y), (x_next+50, y+40), C_PANEL, -1)
            cv2.rectangle(canvas, (x_next, y), (x_next+50, y+40), C_BTN, 1)
            cv2.putText(canvas, ">", (x_next+15, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_BTN, 2)
            
            y += 60
            
            # --- SECTION: GLOBAL STATS ---
            cv2.rectangle(canvas, (x, y), (DEBUG_W-20, y+140), C_PANEL, -1)
            y += 25
            cv2.putText(canvas, "GLOBAL STATUS", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_YELLOW, 1)
            y += 30
            
            lives = env.env.ram[RLConfig.ADDR_LIVES]
            stage = env.env.ram[RLConfig.ADDR_STAGE] # 0-based or 1-based RAW
            
            cv2.putText(canvas, f"Reward: {total_reward:.1f}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GREEN if total_reward>=0 else C_RED, 2); y += 30
            cv2.putText(canvas, f"RAM Stage: {stage}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)
            cv2.putText(canvas, f"Lives: {lives}", (x+200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN, 1); y += 30
            
            # Base Status
            base_ram = env.env.ram[RLConfig.ADDR_BASE]
            latch = getattr(env, 'base_active', False)
            base_dead = (base_ram == 0 and latch)
            base_txt = "DESTROYED" if base_dead else "ACTIVE / SAFE"
            base_col = C_RED if base_dead else C_GREEN
            cv2.putText(canvas, f"Base: {base_txt} (RAM {base_ram})", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_col, 2)
            y += 50
            
            # --- SECTION: GAME OVER CONDITIONS ---
            cv2.rectangle(canvas, (x, y), (DEBUG_W-20, y+180), C_PANEL, -1)
            y += 25
            cv2.putText(canvas, "GAME OVER CONDITIONS (RAW RAM)", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_YELLOW, 1)
            y += 30
            
            # 1. Base RAM (0x68)
            cv2.putText(canvas, f"RAM[0x68] (Base): {base_ram}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_ACCENT, 2)
            cv2.putText(canvas, f"  Latch: {latch}", (x+250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1); y += 25
            
            base_check = "GAME OVER!" if base_dead else "OK"
            base_check_col = C_RED if base_dead else C_GREEN
            cv2.putText(canvas, f"  -> Base Destroyed: {base_check}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_check_col, 1); y += 30
            
            # 2. Lives RAM (0x51)
            cv2.putText(canvas, f"RAM[0x51] (Lives): {lives}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_ACCENT, 2); y += 25
            
            lives_zero = (lives == 0)
            lives_check = "GAME OVER!" if lives_zero else "OK"
            lives_check_col = C_RED if lives_zero else C_GREEN
            cv2.putText(canvas, f"  -> Lives == 0: {lives_check}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lives_check_col, 1); y += 30
            
            # 3. Stage RAM (0x85)
            current_stage_in_ram = env.env.ram[RLConfig.ADDR_STAGE]
            target_stage = getattr(env, 'start_level', None)
            cv2.putText(canvas, f"RAM[0x85] (Stage): {current_stage_in_ram}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_ACCENT, 2)
            cv2.putText(canvas, f"  Target: {target_stage}", (x+250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1); y += 25
            
            level_complete = False
            if target_stage is not None:
                level_complete = (current_stage_in_ram != target_stage)
            level_txt = "LEVEL COMPLETE!" if level_complete else "In Progress"
            level_col = C_GREEN if level_complete else C_GRAY
            cv2.putText(canvas, f"  -> Stage Changed: {level_txt}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, level_col, 1)
            y += 30
            
            
            # --- SECTION: ENEMY TRACKING (THE IMPORTANT PART) ---
            cv2.rectangle(canvas, (x, y), (DEBUG_W-20, y+180), C_PANEL, -1)
            y += 25
            cv2.putText(canvas, "ENEMY TRACKING", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_YELLOW, 1)
            y += 30
            
            ram_kills = env.env.ram[0x19] # Total killed
            ram_spawn = env.env.ram[0x80] # To be spawned?
            ram_screen = env.env.ram[0xA0] # On screen?
            calc_left = 20 - ram_kills
            
            cv2.putText(canvas, f"Kills (0x19): {ram_kills}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_ACCENT, 2); y += 30
            cv2.putText(canvas, f"Calculated (20 - Kills): {calc_left}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1); y += 30
            
            # CONFIRMED: 0x80 is Total Enemies Remaining (Spawn Queue + Active?)
            cv2.putText(canvas, f"TOTAL ENEMIES (0x80): {ram_spawn}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GREEN, 2); y += 30
            
            cv2.putText(canvas, f"On Screen (0xA0): {ram_screen}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BLUE, 1); y += 30
            cv2.putText(canvas, "NOTE: 0x80 drops on Kill or Bomb", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GRAY, 1)
            y += 50
            
            # --- SECTION: DETAILED RAM ---
            cv2.rectangle(canvas, (x, y), (DEBUG_W-20, y+200), C_PANEL, -1)
            y += 25
            cv2.putText(canvas, "DETAILED RAM MAP", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_YELLOW, 1)
            y += 30
            
            px = env.env.ram[RLConfig.ADDR_PLAYER_X]
            py = env.env.ram[RLConfig.ADDR_PLAYER_Y]
            
            def draw_ram(label, addr):
                val = env.env.ram[addr]
                return f"{label} [0x{addr:02X}]: {val}"

            cv2.putText(canvas, draw_ram("Player X", 0x90), (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1)
            cv2.putText(canvas, draw_ram("Player Y", 0x98), (x+250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1); y += 25
            
            cv2.putText(canvas, f"Kills P1 (0x73): {env.env.ram[0x73]}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1); y+=25
            cv2.putText(canvas, f"Kills P2 (0x74 - Unused): {env.env.ram[0x74]}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1); y+=25
            
            # Misc
            cv2.putText(canvas, draw_ram("Timer?", 0x10), (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1); y += 25
            
            y += 40

            # --- SECTION: REWARD LOG ---
            cv2.putText(canvas, "REWARD HISTORY (Scrollable)", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_ACCENT, 1)
            y += 20
            
            # Render all logs (we will crop later)
            for msg, color in list(reward_log)[::-1]:
                cv2.putText(canvas, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y += 25
            
            # --- CROP & SCROLL ---
            global scroll_y, MAX_SCROLL
            MAX_SCROLL = max(0, y - DEBUG_H + 50)
            scroll_y = min(scroll_y, MAX_SCROLL) # Clamp
            
            # Crop viewport
            viewport = canvas[int(scroll_y):int(scroll_y)+DEBUG_H, :]
            
            # Draw Scrollbar
            if MAX_SCROLL > 0:
                bar_h = int((DEBUG_H / y) * DEBUG_H)
                bar_y = int((scroll_y / MAX_SCROLL) * (DEBUG_H - bar_h))
                cv2.rectangle(viewport, (DEBUG_W-10, bar_y), (DEBUG_W, bar_y+bar_h), C_GRAY, -1)

            cv2.imshow("Debug Manager", viewport)

            # FPS Cap
            elapsed = (time.time() - start_time) * 1000
            wait_ms = max(1, int(16 - elapsed)) # 16ms = 60 FPS
            time.sleep(wait_ms / 1000.0)
            
    except KeyboardInterrupt: pass
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
