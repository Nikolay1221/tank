import os
import time
import gym
import numpy as np
import torch as th
import json  # For metrics persistence
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from stable_baselines3.common.callbacks import CheckpointCallback

from python.config import RLConfig
from python.battle_city_env import BattleCityEnv

from stable_baselines3.common.env_util import make_vec_env
import cv2

METRICS_FILE = './checkpoints/training_metrics.json'

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=1000, save_interval=10000):
        super().__init__(verbose)
        self.episode_rewards = deque(maxlen=100)
        self.episode_kills = deque(maxlen=100)
        self.total_games = 0
        self.log_interval = log_interval
        self.save_interval = save_interval  # Save metrics every N steps
        self.step_count = 0

    def save_metrics(self):
        """Save training metrics to JSON file."""
        metrics = {
            'total_games': int(self.total_games),
            'step_count': int(self.step_count),
            'episode_kills': [int(k) for k in self.episode_kills],
            'episode_rewards': [float(r) for r in self.episode_rewards],
        }
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[METRICS SAVED] Games: {self.total_games}, Steps: {self.step_count}")

    def load_metrics(self):
        """Load training metrics from JSON file."""
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, 'r') as f:
                    metrics = json.load(f)
                self.total_games = metrics.get('total_games', 0)
                self.step_count = metrics.get('step_count', 0)
                self.episode_kills = deque(metrics.get('episode_kills', []), maxlen=100)
                self.episode_rewards = deque(metrics.get('episode_rewards', []), maxlen=100)
                print(f"[METRICS LOADED] Games: {self.total_games}, Steps: {self.step_count}")
                return True
            except Exception as e:
                print(f"[METRICS LOAD FAILED] {e}")
        return False

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Check for episode end
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.total_games += 1
                info = self.locals['infos'][i]
                
                if 'kills' in info:
                    self.episode_kills.append(info['kills'])
                
                if self.verbose > 0:
                     kill_avg = np.mean(self.episode_kills) if self.episode_kills else 0
                     self.logger.record("custom/games_played", self.total_games)
                     self.logger.record("custom/avg_kills", kill_avg)
                     print(f"[EPISODE END] Game #{self.total_games}, Kills: {info.get('kills', 0)}, Avg Kills: {kill_avg:.2f}")
        
        # PERIODIC LOG - every log_interval steps
        if self.step_count % self.log_interval == 0:
            kill_avg = np.mean(self.episode_kills) if self.episode_kills else 0
            print(f"[STEP {self.step_count}] Games: {self.total_games}, Avg Kills: {kill_avg:.2f}, Buffer: {len(self.episode_kills)} episodes")
        
        # AUTO-SAVE METRICS - every save_interval steps (sync with checkpoint)
        if self.step_count % self.save_interval == 0:
            self.save_metrics()
        
        # Explicit rendering in Main Process
        if not RLConfig.HEADLESS:
            try:
                # Render using the original env logic (which returns RGB array)
                frames = self.training_env.render(mode='rgb_array') 
                
                if frames is not None:
                    target_frame = None
                    if isinstance(frames, list):
                        target_frame = frames[0]
                    elif isinstance(frames, np.ndarray):
                            if len(frames.shape) == 4: # (N, H, W, C)
                                target_frame = frames[0]
                            else:
                                target_frame = frames
                    
                    if target_frame is not None:
                            # Convert RGB to BGR
                            frame_bgr = cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR)
                            # Resize
                            frame_bgr = cv2.resize(frame_bgr, (512, 480), interpolation=cv2.INTER_NEAREST)
                            cv2.imshow("Battle City AI", frame_bgr)
                            cv2.waitKey(1)
            except Exception as e:
                pass
                        
        return True

def make_env_rank(rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Distribute levels 1-35 across environments
        if RLConfig.TRAIN_SINGLE_LEVEL is not None:
            level = RLConfig.TRAIN_SINGLE_LEVEL
        else:
            level = (rank % 35) + 1
        
        env = BattleCityEnv(render_mode='rgb_array', start_level=level)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    print(f"Starting training with {RLConfig.N_ENVS} environments...")
    print(f"Game Path: {RLConfig.GAME_PATH}")
    print(f"Algorithm: PPO (Standard CNN)")
    print(f"Headless Mode: {RLConfig.HEADLESS}")

    # Create the vectorized environment
    env_fns = [make_env_rank(i) for i in range(RLConfig.N_ENVS)]
    
    if RLConfig.N_ENVS > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    
    # WRAPPER: Frame Stacking (Crucial for Visual RL)
    # Stacks 4 frames on channel dimension. Input: (84, 84, 1) -> (84, 84, 4)
    # Even with LSTM, frame stacking is beneficial for immediate motion detection.
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    # WRAPPER: Normalize Rewards (v.v. IMPORTANT for PPO)
    # Scales large rewards (+100, +400) to standard Gaussian range approx [-1, 1].
    # Keeps gamma=0.99 by default.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    # Monitor (logs)
    vec_env = VecMonitor(vec_env, "logs/TestMonitor")

    # Policy Architecture: CnnPolicy
    # "The Monster Brain" - Massive Deep Funnel
    # Input (3136) -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> Heads
    massive_arch = [2048, 1024, 512, 256, 128, 64, 32, 16]
    policy_kwargs = dict(
        net_arch=dict(pi=massive_arch, vf=massive_arch)
    )
    
    CHECKPOINT_DIR = './checkpoints/'
    latest_checkpoint = None
    
    if os.path.exists(CHECKPOINT_DIR):
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('battle_city_ppo') and f.endswith('.zip') and 'final' not in f]
        if files:
            def get_step_count(filename):
                parts = filename.split('_')
                if 'steps.zip' in filename:
                    try:
                        return int(parts[-2])
                    except:
                        return 0
                return 0
            
            # Sort by steps (descending)
            files.sort(key=get_step_count, reverse=True)
            candidate = files[0]
            if get_step_count(candidate) > 0:
                latest_checkpoint = os.path.join(CHECKPOINT_DIR, candidate)
                print(f"RESUMING from checkpoint: {latest_checkpoint}")
    
    if latest_checkpoint:
        try:
            print("Attempting to load (seamless resume)...")
            model = PPO.load(
                latest_checkpoint, 
                env=vec_env,
                verbose=1,
                # NO explicit params = use everything from checkpoint as-is
                tensorboard_log="./tensorboard_logs/",
                device="cuda"
            )
        except Exception as e:
             print(f"Checkpoint load failed (Likely Architecture Mismatch): {e}")
             print("Starting FRESH training (Architecture Change detected)...")
             model = None
    
    if latest_checkpoint is None or 'model' not in locals() or model is None:
        checkpoint_dir = './checkpoints/'
        curr_run_id = 1
        while os.path.exists(f"{checkpoint_dir}/run_{curr_run_id}"):
            curr_run_id += 1
        
        run_dir = f"{checkpoint_dir}/run_{curr_run_id}"
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"Start Training Run #{curr_run_id}...")
        
        # Standard PPO with CnnPolicy
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            learning_rate=RLConfig.LEARNING_RATE,  # Constant LR (no schedule)
            n_steps=RLConfig.N_STEPS,
            batch_size=RLConfig.BATCH_SIZE,
            n_epochs=10, # User requested revert to 10 (Maximum learning per batch)
            ent_coef=RLConfig.ENTROPY_COEF,
            tensorboard_log="./tensorboard_logs/",
            policy_kwargs=policy_kwargs,
            device="cuda" 
        )

    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./checkpoints/',
        name_prefix='battle_city_ppo' # Keep same prefix? Or change to battle_city_lstm? Keeping same allows mixup but clearer history.
    )
    
    logging_callback = LoggingCallback(verbose=1, save_interval=10000)
    
    # LOAD METRICS if resuming
    is_resuming = latest_checkpoint is not None and model is not None # Only load if model loaded successfully
    if is_resuming:
        logging_callback.load_metrics()

    try:
        model.learn(
            total_timesteps=RLConfig.TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, logging_callback],
            reset_num_timesteps=not is_resuming  # False if resuming = continue count
        )
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        # SAVE METRICS before exit
        logging_callback.save_metrics()
        model.save("battle_city_ppo_final")
        vec_env.close()
        print("Model and metrics saved. Environments closed.")

if __name__ == "__main__":
    main()
