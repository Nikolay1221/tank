import gym
from gym import spaces
import numpy as np
import cv2
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace
from .config import RLConfig

# Define simple actions: Move and Fire
# Battle City controls: arrows for movement, A/B for fire.
# We'll combine movement + fire for efficiency.
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['up'],
    ['right'],
    ['down'],
    ['left'],
    ['A'],
    ['up', 'A'],
    ['right', 'A'],
    ['down', 'A'],
    ['left', 'A'],
]

class BattleCityEnv(gym.Wrapper):
    def __init__(self, render_mode=None, is_visible=False, start_level=None):
        # Create base NES environment
        env = NESEnv(RLConfig.GAME_PATH)
        
        # Apply Joypad wrapper to discrete actions
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        

        
        # Use gym.Wrapper
        super().__init__(env)
        
        # Store render_mode locally. 
        # We don't set it on env because JoypadSpace might block it (property)
        
        # Store render_mode locally. 
        self._custom_render_mode = render_mode
        self.is_visible = is_visible
        self.start_level = start_level
        self.metadata = {'render_modes': ['human', 'rgb_array']}
        
        # Observation Space: 84x84 Grayscale (Standard Atari)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
        # Trackers
        self.prev_kills = [0] * 4
        self.prev_lives = 0
        
    def _get_obs(self):
        """Returns the current screen, grayscale, resized to 84x84."""
        # Get raw screen from nes_py (240, 256, 3)
        screen = self.env.screen.copy()
        
        # Convert to Grayscale
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84 (Standard Nature CNN input)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Add channel dimension (84, 84, 1)
        return np.expand_dims(resized, axis=-1)

    def reset(self, **kwargs):
        # Handle Gymnasium vs Gym API differences
        # nes-py doesn't accept 'seed' or 'options' in reset()
        if 'seed' in kwargs:
            # We can use the seed to seed numpy if we want, but NESEnv might not support it directly
            pass 
        kwargs.pop('seed', None)
        kwargs.pop('options', None)

        # We ignore the pixel observation returned by reset
        _ = self.env.reset(**kwargs)
        
        # Skip Title Screen (Full 5-step sequence)
        
        # 1. Wait 80 frames (Logo music)
        # print("Reset: Boot Sequence Step 1/5...")
        for _ in range(80): 
            self.env.env.step(0) 
            
        # 2. Press START (Skip Title)
        # print("Reset: Boot Sequence Step 2/5 (Skip Title)...")
        for _ in range(10): self.env.env.step(8) 
        for _ in range(30): self.env.env.step(0) 

        # Force Level Selection if specified
        if self.start_level is not None:
             # Level is stored at 0x85. Note: Internal levels likely 0-34? 
             # Or 1-35? Usually 0-based in RAM, but visual is 1-based.
             # Assuming 1-based input, so we write start_level - 1.
             # Need to be sure when to write this. 
             # Writing it BEFORE hitting start on the "Stage 1" screen is safest.
             
             # Let's write it now, before we select 1 player and before the stage screen.
             # Just to be safe, write it multiple times or ensure it sticks.
             target_val = self.start_level # RAM 1 = Stage 1 (User confirmed)
             self.env.ram[RLConfig.ADDR_STAGE] = target_val
        
        # 3. Press START (Select 1 Player)
        # print("Reset: Boot Sequence Step 3/5 (Select 1 Player)...")
        for _ in range(10): self.env.env.step(8)
        for _ in range(30): self.env.env.step(0)

        # Re-enforce Level Selection just in case
        if self.start_level is not None:
             target_val = self.start_level
             self.env.ram[RLConfig.ADDR_STAGE] = target_val

        # 4. Press START (Skip STAGE 1 Screen)
        # print("Reset: Boot Sequence Step 4/5 (Skip Stage Screen)...")
        for _ in range(10): self.env.env.step(8)
        
        # 5. Wait 60 frames (Curtain open)
        # print("Reset: Boot Sequence Step 5/5 (Wait Curtain)...")
        for _ in range(60): self.env.env.step(0)
             
        # Reset trackers
        self.prev_kills = [0] * 4
        self.death_count = 0
        # Initialize lives to ACTUAL value from RAM
        self.prev_lives = self.env.ram[0x51]
        
        # Reset Proximity Reward Tracker
        self.prev_min_dist = self._get_nearest_dist()
        self.prev_px = self.env.ram[RLConfig.ADDR_PLAYER_X]
        self.prev_py = self.env.ram[RLConfig.ADDR_PLAYER_Y]
        
        # Exploration Tracker
        self.visited_cells = set()

        # Proximity Reward Logic
        self.prev_min_dist = 999.0
        self.prev_px = 0
        self.prev_py = 0
        
        # Base Latch: 
        # RAM[0x68] is 0 at boot, 80 when active, 0 when destroyed.
        # We must wait for it to become non-zero (active) before checking for 0 (destroyed).
        self.base_active = False 
        
        # Load Game Over template (REMOVED - Using RAM Logic)
        self.game_over_template = None
            
        return self._get_obs(), {}

    def _get_nearest_dist(self):
        """Calculates distance to nearest active enemy."""
        # Player coordinates (Center)
        # RAM gives Top-Left. Sprite is 16x16. Center is +8.
        px = float(self.env.ram[RLConfig.ADDR_PLAYER_X]) + 8.0
        py = float(self.env.ram[RLConfig.ADDR_PLAYER_Y]) + 8.0
        
        min_dist = 999.0
        found = False

        # Iterate Enemy Slots 2 to 7
        for i in range(2, 8):
            # Check Status (Alive >= 128)
            status = self.env.ram[RLConfig.ADDR_ENEMY_STATUS_BASE + i]
            if status < 128:
                continue
            
            # Enemy Coordinates
            ex = float(self.env.ram[RLConfig.ADDR_COORD_X_BASE + i]) + 8.0
            ey = float(self.env.ram[RLConfig.ADDR_COORD_Y_BASE + i]) + 8.0
            
            # Simple 0,0 check (sometimes happens during init)
            if self.env.ram[RLConfig.ADDR_COORD_X_BASE + i] == 0 and self.env.ram[RLConfig.ADDR_COORD_Y_BASE + i] == 0:
                continue
            
            # Euclidian Distance
            d = np.sqrt((ex - px)**2 + (ey - py)**2)
            if d < min_dist:
                min_dist = d
                found = True
                
        return min_dist if found else 999.0

    def _check_game_over(self):
        if self.game_over_template is None:
            return False
            
        import cv2
        # Get frame in BGR (nes-py render is RGB)
        # NES native resolution is 256x240
        frame_rgb = self.env.render(mode='rgb_array')
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Safety Check: Dimensions
        # If template is larger than frame (e.g. 512x480 vs 256x240), we must downscale template
        fh, fw = frame_bgr.shape[:2]
        th, tw = self.game_over_template.shape[:2]
        
        template_to_use = self.game_over_template
        if th > fh or tw > fw:
             # Assuming 2x scale difference (common)
             scale_x = fw / tw
             scale_y = fh / th
             scale = min(scale_x, scale_y)
             new_w = int(tw * scale)
             new_h = int(th * scale)
             template_to_use = cv2.resize(self.game_over_template, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Template Matching
        try:
            res = cv2.matchTemplate(frame_bgr, template_to_use, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Threshold 0.75 (slightly lower for robustness)
            if max_val > 0.75:
                return True
        except Exception as e:
            # print(f"Game Over Check Error: {e}")
            pass
            
        return False

    def step(self, action):
        # We ignore the pixel observation returned by step
        _, reward, done, info = self.env.step(action)
        
        # Check for Visual Game Over
        visual_penalty = 0
        if not done:
             # OpenCV Check (Backup or Removed)
             # if self._check_game_over():
             #     done = True
             #     info['visual_game_over'] = True
             #     visual_penalty = -20
             
             # RAM Check for Base Destruction (0x68)
             # Logic: 0 (Boot) -> 80 (Active) -> 0 (Destroyed)
             base_status = self.env.ram[RLConfig.ADDR_BASE]
             
             if not self.base_active:
                 if base_status != 0:
                     self.base_active = True
             else:
                 # Base was active, now checking if destroyed
                 if base_status == 0:
                     done = True
                     info['base_destroyed'] = True
                     visual_penalty = -10 # Increased to -10 as requested
        
        # Check for Lives (0x51)
        # If lives == 0, it means Game Over (or about to be)
        if not done:
            curr_lives = self.env.ram[RLConfig.ADDR_LIVES]
            if curr_lives == 0:
                done = True
                info['game_over'] = True
                info['game_over'] = True
                visual_penalty = -20 # Penalty for losing all lives (Same as Base Destruction)
        
        # Check for Level Completion / Stage Change
        # If we are enforcing a specific level, we must NOT allow the game to proceed to the next stage.
        # If the stage in RAM changes, it means the current level was beaten.
        if self.start_level is not None:
             current_stage_ram = self.env.ram[RLConfig.ADDR_STAGE]
             # Assuming start_level is 1-based, RAM is usually 0-based or 1-based.
             # We injected (start_level).
             # If it changes, end the episode.
             if current_stage_ram != self.start_level:
                 done = True
                 info['is_success'] = True
                 # Optional: Large reward for clearing level?
                 # custom_reward += 50 
        
        # Custom Reward Logic
        curr_kills = [self.env.ram[addr] for addr in RLConfig.ADDR_KILLS]
        curr_lives = self.env.ram[RLConfig.ADDR_LIVES]
        
        # Calculate kill reward
        kill_reward = 0
        # Cast to int to prevent numpy uint8 overflow
        total_kills = int(sum(curr_kills))
        total_prev_kills = int(sum(self.prev_kills))
        
        # Calculate new kills in this step
        new_kills = total_kills - total_prev_kills
        
        if new_kills > 0:
            for k in range(new_kills):
                # Kill index in the current session (1-based)
                kill_idx = total_prev_kills + k + 1
                
                # Base reward 5. Additional reward starts from 2nd kill.
                # 1st kill: 5 + 0 = 5
                # 2nd kill: 5 + 2 = 7
                # 3rd kill: 5 + 3 = 8
                # ...
                bonus = kill_idx if kill_idx > 1 else 0
                kill_reward += 5 + bonus
                
        # HUGE BONUS for Level Completion (20 Kills)
        # Assuming typical level has 20 enemies.
        if total_kills >= 20 and sum(self.prev_kills) < 20:
             info['is_success'] = True
             kill_reward += 100 # JACKPOT
        
        # Calculate death penalty
        death_penalty = 0
        if curr_lives < self.prev_lives:
            self.death_count += 1
            death_penalty = -self.death_count
            
        # EXPLORATION REWARD
        # Map is 192x192 pixels (from 24 to 216).
        # We divide it into 16x16 px cells. Total 12x12 = 144 cells.
        exploration_reward = 0
        
        # Read raw coordinates
        px = self.env.ram[RLConfig.ADDR_PLAYER_X]
        py = self.env.ram[RLConfig.ADDR_PLAYER_Y]
        
        # Bounds checks
        if 24 <= px <= 216 and 24 <= py <= 216:
            # 16px grid
            grid_x = (px - 24) // 16
            grid_y = (py - 24) // 16
            
            cell_id = (grid_x, grid_y)
            
            if cell_id not in self.visited_cells:
                self.visited_cells.add(cell_id)
                exploration_reward = 0.0 # Disabled as requested (was 1.0)
        
        # PROXIMITY REWARD (Approaching Enemies)
        proximity_reward = 0.0
        
        # Current Player Position
        curr_px = self.env.ram[RLConfig.ADDR_PLAYER_X]
        curr_py = self.env.ram[RLConfig.ADDR_PLAYER_Y]
        
        # Check if player moved (Anti-Camping)
        # We only reward proximity if the player is actively moving.
        player_moved = (curr_px != self.prev_px) or (curr_py != self.prev_py)
        
        # 1. Get current distance
        curr_dist = self._get_nearest_dist()
        
        # 2. Validation
        if curr_dist != 999.0 and self.prev_min_dist != 999.0:
            # 3. Calculate Difference
            # Positive diff = We got closer (Good)
            diff = self.prev_min_dist - curr_dist
            
            # 4. Filters for Teleportation / Respawn
            if abs(diff) <= 50.0:
                 # Anti-Camping Logic:
                 if player_moved:
                     proximity_reward = diff * 0.5 # Reduced from 2.0 to 0.5
                 else:
                     # If not moved, reward is 0 (ignore enemy approach or retreat)
                     proximity_reward = 0.0
        
        # 5. Update state
        self.prev_min_dist = curr_dist
        self.prev_px = curr_px
        self.prev_py = curr_py
            
        # Total reward
        custom_reward = kill_reward + death_penalty + visual_penalty + exploration_reward + proximity_reward
        
        # Update trackers
        self.prev_kills = curr_kills
        self.prev_lives = curr_lives
        
        # Info for debugging
        info['kills'] = sum(curr_kills)
        info['lives'] = curr_lives
        info['lives'] = curr_lives
        info['exploration'] = len(self.visited_cells)
        info['proximity_reward'] = proximity_reward
        
        truncated = False
        
        return self._get_obs(), custom_reward, done, truncated, info
        
    @property
    def render_mode(self):
        return self._custom_render_mode

    def render(self):
        # Explicit render handling
        # if self.is_visible: ... removed for headless optimization
             
        # Also return frame if needed by callbacks (so VideoRecorder works if we use it)
        if self._custom_render_mode == 'rgb_array':
             return self.env.render(mode='rgb_array')

