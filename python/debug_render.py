from python.battle_city_env import BattleCityEnv
import time

def main():
    print("Creating environment...")
    env = BattleCityEnv()
    env.reset()
    print("Environment created and reset.")
    
    print("Starting rendering loop...")
    try:
        for i in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.016) # ~60 FPS
            
            if terminated or truncated:
                env.reset()
                
        print("Loop finished.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main()
