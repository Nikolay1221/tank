import os

class RLConfig:
    # Environment settings
    GAME_PATH = os.path.abspath("BattleCity.nes")
    SKIP_FRAMES = 4 # Standard for Atari/Visual (Capture motion)
    
    # Training settings
    # 60 FPS / 4 = 15 steps/sec. 
    # 10M frames / 4 = 2.5M steps.
    TOTAL_TIMESTEPS = 240000000  
    
    # Multiprocessing: 70 environments (2 per level) for server
    # 35 Levels * 2 Agents = 70 Envs
    N_ENVS = 8
    
    # Level Selection
    # Valid values: 1-35 (Train on specific level) or None (Train on all levels)
    TRAIN_SINGLE_LEVEL = 1   
    
    # Model settings
    LEARNING_RATE = 2.5e-4 # Reduced 10x for Fine-Tuning (was 2.5e-4)
    ENTROPY_COEF = 0.01
    
    # N_STEPS: Standard PPO
    N_STEPS = 2048 
    
  
    # Tesla T4 переварит это мгновенно. 
    BATCH_SIZE = 4096
    
    # Display settings
    HEADLESS = False  # На сервере ВСЕГДА True. Иначе крашнется.
    RENDER_RANK_0 = False # На сервере выключай, смотреть некому.
    
    # RAM Addresses (Тут все ок)
    ADDR_KILLS = [0x73, 0x74, 0x75, 0x76]
    ADDR_LIVES = 0x51
    ADDR_STAGE = 0x85
    ADDR_BASE = 0x68 # 0=Alive, Non-Zero=Destroyed (Hypothesis from Game Genie GXUTATSA)
    ADDR_PLAYER_X = 0x90
    ADDR_PLAYER_Y = 0x98
    
    @staticmethod
    def get_rom_path():
        if not os.path.exists(RLConfig.GAME_PATH):
             raise FileNotFoundError(f"ROM not found at {RLConfig.GAME_PATH}")
        return RLConfig.GAME_PATH
