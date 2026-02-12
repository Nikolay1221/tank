import json
import matplotlib.pyplot as plt
import numpy as np
import os

METRICS_FILE = './checkpoints/training_metrics.json'
OUTPUT_IMAGE = 'training_results.png'

def moving_average(data, window_size=50):
    """Calculates moving average for smoother plots."""
    if len(data) < window_size:
        return data
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def main():
    if not os.path.exists(METRICS_FILE):
        print(f"Error: Metrics file not found at {METRICS_FILE}")
        return

    print(f"Loading metrics from {METRICS_FILE}...")
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)

    kills = metrics.get('all_episode_kills', [])
    rewards = metrics.get('all_episode_rewards', [])
    total_games = len(kills)

    if total_games == 0:
        print("No game history found in metrics.")
        return

    print(f"Found history for {total_games} games.")

    # Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Kills per Game
    x = np.arange(1, total_games + 1)
    ax1.plot(x, kills, label='Kills', color='blue', alpha=0.3, linewidth=1)
    
    # Moving Average for Kills
    window = max(10, total_games // 20) # Adaptive window
    avg_kills = moving_average(kills, window)
    if len(avg_kills) > 0:
        # Pad beginning to align x-axis
        pad = len(x) - len(avg_kills)
        ax1.plot(x[pad:], avg_kills, label=f'Avg Kills ({window} games)', color='red', linewidth=2)
    
    ax1.set_ylabel('Kills per Game')
    ax1.set_title(f'Kills History (Total Games: {total_games})')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Reward per Game (if available)
    if rewards and len(rewards) == total_games:
        ax2.plot(x, rewards, label='Reward', color='green', alpha=0.3, linewidth=1)
        
        avg_rewards = moving_average(rewards, window)
        if len(avg_rewards) > 0:
             pad = len(x) - len(avg_rewards)
             ax2.plot(x[pad:], avg_rewards, label=f'Avg Reward ({window} games)', color='orange', linewidth=2)
        
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Reward History')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Reward History Available', ha='center', va='center')

    ax2.set_xlabel('Game Number (Episode)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Plot saved to {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    main()
