from python.battle_city_env import BattleCityEnv
import numpy as np
import time

def verify_rewards():
    print("Starting Reward Verification...")
    env = BattleCityEnv(render_mode='rgb_array', is_visible=False)
    obs, info = env.reset()
    
    rewards = []
    
    # Run for 500 steps
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        if done:
            obs, info = env.reset()
            
    rewards = np.array(rewards)
    print(f"Total Steps: {len(rewards)}")
    print(f"Min Reward: {rewards.min()}")
    print(f"Max Reward: {rewards.max()}")
    print(f"Mean Reward: {rewards.mean()}")
    
    if rewards.max() > 1.0 or rewards.min() < -1.0:
        print("WARNING: Rewards are outside [-1, 1] range!")
    else:
        print("SUCCESS: Rewards are within [-1, 1] range.")

if __name__ == "__main__":
    verify_rewards()
