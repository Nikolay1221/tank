import os
import time
import numpy as np
import torch as th
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from python.config import RLConfig
from python.battle_city_env import BattleCityEnv

def main():
    print("Searching for model...")
    checkpoint_dir = "./checkpoints/"
    final_model = "battle_city_ppo_final.zip"
    
    model_path = None

    # 1. Try final model in root
    if os.path.exists(final_model):
        print(f"Found FINAL model: {final_model}")
        model_path = final_model
    else:
        # 2. Try latest checkpoint
        if not os.path.exists(checkpoint_dir):
            print(f"Directory {checkpoint_dir} not found!")
            return

        # Find all zip files
        files = [f for f in os.listdir(checkpoint_dir) if f.startswith('battle_city_ppo') and f.endswith('.zip')]
        if not files:
            print("No checkpoints found!")
            return

        # Sort by modification time to get the ABSOLUTE LATEST one regardless of step count in name
        files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        
        latest_file = files[0]
        print(f"Loading LATEST checkpoint (by time): {latest_file}...")
        model_path = os.path.join(checkpoint_dir, latest_file)
    
    # Create Env
    env = DummyVecEnv([lambda: BattleCityEnv(render_mode='rgb_array', start_level=1)])
    
    # WRAPPER: Frame Stacking (Match Training!)
    env = VecFrameStack(env, n_stack=4)
    
    # Load Model (PPO)
    print(f"Loading weights from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    print("\n--- STARTING PLAYBACK ---")
    print("Mode: STOCHASTIC")
    print("Press 'S' to toggle Deterministic mode (currently OFF)")
    print("-------------------------")
    
    obs = env.reset()
    deterministic = False
    
    try:
        while True:
            # Predict action
            action, _ = model.predict(
                obs, 
                deterministic=deterministic
            )
            
            # Step
            obs, rewards, dones, infos = env.step(action)
            
            frame = env.render(mode='rgb_array') 
            
            if frame is not None:
                if isinstance(frame, list): frame = frame[0]
                if len(frame.shape) == 4: frame = frame[0]
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (512, 480), interpolation=cv2.INTER_NEAREST)
                
                # Add overlay text
                mode_text = "DETERMINISTIC" if deterministic else "STOCHASTIC"
                cv2.putText(frame, f"Mode: {mode_text} (Press S to toggle)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Battle City AI - Watch Mode", frame)
                key = cv2.waitKey(15) & 0xFF
                if key == ord('s'):
                    deterministic = not deterministic
                    print(f"Switched to {'DETERMINISTIC' if deterministic else 'STOCHASTIC'} mode")
                elif key == ord('q'):
                    break
            
            if dones[0]:
                print(f"Episode Done. Kills: {infos[0].get('kills', 0)}")
                # Obs is automatically reset
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cv2.destroyAllWindows()
        env.close()

if __name__ == "__main__":
    main()