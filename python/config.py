import os

class RLConfig:
    GAME_PATH = os.path.abspath("BattleCity.nes")
    SKIP_FRAMES = 20
    
    TOTAL_TIMESTEPS = 240000000  
    N_ENVS = 16
    TRAIN_SINGLE_LEVEL = 1   
    INVULNERABLE_BASE = True
    
    # Стандартный Atari PPO LR (пониженный для стабильности)
    LEARNING_RATE = 1.0e-4 
    ENTROPY_COEF = 0.1
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    
    # Target KL new parameter (строгий, чтобы не ломать веса)
    TARGET_KL = 0.02
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    # Широкая архитектура (без узкого горлышка)
    HIDDEN_LAYERS = [4096, 4096, 4096, 2048]
    CNN_FEATURES_DIM = 4096
    
    # NES-recommended
    N_STEPS = 512
    BATCH_SIZE = 512
    N_EPOCHS = 4
    
    SAVE_FREQ = 10000
    LOG_INTERVAL = 1000
    
    HEADLESS = False
    RENDER_RANK_0 = False
    
    ADDR_KILLS = [0x73, 0x74, 0x75, 0x76]
    ADDR_LIVES = 0x51
    ADDR_STAGE = 0x85
    ADDR_BASE = 0x68
    ADDR_PLAYER_X = 0x90
    ADDR_PLAYER_Y = 0x98
    
    ADDR_ENEMY_STATUS_BASE = 0xA0
    ADDR_COORD_X_BASE = 0x90
    ADDR_COORD_Y_BASE = 0x98
    
    @staticmethod
    def get_rom_path():
        if not os.path.exists(RLConfig.GAME_PATH):
             raise FileNotFoundError(f"ROM not found at {RLConfig.GAME_PATH}")
        return RLConfig.GAME_PATH
