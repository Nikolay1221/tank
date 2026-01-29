#!/bin/bash
set -e # Exit on error

echo ">>> 1. Adding Python 3.10 Repo (Deadsnakes)..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq

echo ">>> 2. Installing Python 3.10 & Deps..."
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip build-essential libsdl2-dev libgl1 libglib2.0-0 unzip

echo ">>> 3. DELETING old venv and creating fresh one..."
sudo rm -rf venv
python3.10 -m venv venv
echo "    Venv created with Python 3.10."

# Activate
source venv/bin/activate

echo ">>> 4. Installing NumPy < 2 FIRST (fixes nes-py compatibility)..."
pip install --upgrade pip
pip install 'numpy<2'

echo ">>> 5. Installing other requirements..."
pip install -r requirements.txt
pip install sb3-contrib

echo ">>> 6. Updating & Restarting Services..."
sudo cp linux/battle_city_train.service /etc/systemd/system/
sudo cp linux/battle_city_board.service /etc/systemd/system/
sudo systemctl daemon-reload

sudo systemctl enable battle_city_board
sudo systemctl enable battle_city_train
sudo systemctl restart battle_city_board
sudo systemctl restart battle_city_train

echo ">>> DONE! Services restarted using Python 3.10 + NumPy 1.x"
echo "    Check status with: sudo systemctl status battle_city_train"
