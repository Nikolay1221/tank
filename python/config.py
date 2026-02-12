import os

class RLConfig:
    GAME_PATH = os.path.abspath("BattleCity.nes")
    SKIP_FRAMES = 2
    
    TOTAL_TIMESTEPS = 240000000  
    N_ENVS = 16
    TRAIN_SINGLE_LEVEL = 1   
    
    # Снижаем LR для стабильности гигантской сети
    LEARNING_RATE = 1.0e-4 
    ENTROPY_COEF = 0.01
    GAMMA = 0.99
    GAE_LAMBDA = 0.90
    CLIP_RANGE = 0.2
    
    # Target KL new parameter
    TARGET_KL = 0.03
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    # Ультра-частые обновления (как просил)
    N_STEPS = 512 
    BATCH_SIZE = 512
    N_EPOCHS = 7
    
    # Гигантская и глубокая сеть (3 слоя по 2048 нейронов)
    HIDDEN_LAYERS = [2048,2048,2048]
    
    SAVE_FREQ = 10000
    LOG_INTERVAL = 1000
    
    HEADLESS = True
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
