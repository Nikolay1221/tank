
import os
import sys

print("--- PYTHON ENVIRONMENT CHECK ---")
print(f"Python: {sys.version}")

try:
    import cv2
    print(f"OpenCV: OK {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV: FAILED ({e})")

try:
    import gym
    print(f"Gym: OK {gym.__version__}")
except ImportError as e:
    print(f"Gym: FAILED ({e})")

try:
    import stable_baselines3
    print(f"SB3: OK {stable_baselines3.__version__}")
except ImportError as e:
    print(f"SB3: FAILED ({e})")

print("\n--- CHECKING LOGS FOR ERRORS ---")
log_path = "/var/log/battle_city_train.log"
if os.path.exists(log_path):
    with open(log_path, 'r', errors='ignore') as f:
        lines = f.readlines()
        
    print(f"Log has {len(lines)} lines.")
    print("Listing lines containing 'Traceback' or 'Error':")
    
    found = False
    for i, line in enumerate(lines):
        if "Traceback" in line or "Error" in line:
            print(f"Line {i}: {line.strip()}")
            found = True
            # Print context (next 10 lines) if Traceback found
            if "Traceback" in line:
                for j in range(1, 15):
                    if i+j < len(lines):
                        print(f"    {lines[i+j].strip()}")
    
    if not found:
        print("No obvious errors found in log. Printing last 20 lines:")
        for line in lines[-20:]:
            print(line.strip())
else:
    print("Log file not found!")
