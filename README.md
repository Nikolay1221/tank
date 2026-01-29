# Battle City AI (RL)

This project trains a Reinforcement Learning agent to play Battle City (NES) using PPO.

## Structure
- `python/config.py`: Configuration (Hyperparameters, Paths, Headless mode).
- `python/battle_city_env.py`: Custom Gymnasium Environment (Reward shaping).
- `python/train.py`: Main training script with multiprocessing.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure `BattleCity.nes` is in the root directory.

## Usage
Run the training script:
```bash
python -m python.train
```

## Configuration
Edit `python/config.py` to change:
- `HEADLESS`: `True` (background) or `False` (render rank 0).
- `N_ENVS`: Number of parallel environments.
