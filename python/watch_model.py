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
    print("Searching for latest checkpoint...")
    checkpoint_dir = "./checkpoints/"
    
    # Check if dir exists
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} not found!")
        return

    # Find all zip files
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('battle_city_ppo') and f.endswith('.zip')]
    if not files:
        print("No checkpoints found!")
        return

    # Sort by step count (extract number from battle_city_ppo_XXXXXX_steps.zip)
    def get_steps(filename):
        try:
            return int(filename.split('_')[-2])
        except:
            return 0
    
    latest_file = max(files, key=get_steps)
    print(f"Loading Model: {latest_file}...")
    
    # Create Env
    # Use DummyVecEnv for single-threaded viewing
    # MUST use 'rgb_array' so automatic render() returns the frame we can show
    env = DummyVecEnv([lambda: BattleCityEnv(render_mode='rgb_array', start_level=2)])
    
    # WRAPPER: Frame Stacking (Match Training!)
    env = VecFrameStack(env, n_stack=4)
    
    # Load Model (PPO)
    model = PPO.load(os.path.join(checkpoint_dir, latest_file), device="cpu")
    
    print("\n--- STARTING PLAYBACK ---")
    print("Mode: STOCHASTIC (Randomness Enabled)")
    print("Matches Training Behavior")
    print("-------------------------")
    
    obs = env.reset()
    
    try:
        while True:
            # Predict action
            # PPO is stateless (no LSTM state)
            action, _ = model.predict(
                obs, 
                deterministic=True
            )
            
            # Step
            obs, rewards, dones, infos = env.step(action)
            
            # Render Window
            # Env is wrapped in VecFrameStack -> DummyVecEnv
            # But render() calls the base envs.
            
            # Get original env to render
            # raw_env = env.envs[0] # This worked for DummyVecEnv
            # But VecFrameStack wraps it. 
            # We can just call env.render()? SB3 VecEnv render might differ.
            # Best to access the underlying env.
            
            frame = env.render(mode='rgb_array') # VecFrameStack usually passes this down
            
            if frame is not None:
                # If frame is a list (from VecEnv), take first
                if isinstance(frame, list): frame = frame[0]
                # If frame is (1, H, W, C) or (H, W, C)
                if len(frame.shape) == 4: frame = frame[0]
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (512, 480), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Battle City AI - Watch Mode", frame)
                cv2.waitKey(20) # ~ 50 FPS
            
            if dones[0]:
                print(f"Episode Done. Reward: {infos[0].get('episode', {}).get('r', 'N/A')}")
                # Obs is automatically reset
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
