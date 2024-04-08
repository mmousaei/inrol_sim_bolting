import numpy as np
from scipy.spatial.transform import Rotation as R
import random

def interpolate_angles(start_angle, end_angle, steps):
    start_angle = (start_angle + 180) % 360 - 180
    end_angle = (end_angle + 180) % 360 - 180
    delta = (end_angle - start_angle) % 360
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360
    interpolated = [start_angle + (delta * step / steps) for step in range(steps)]
    normalized = [(angle + 180) % 360 - 180 for angle in interpolated]
    return normalized

def interpolate_positions(start_pos, end_pos, steps):
    interpolated_positions = [start_pos + (end_pos - start_pos) * (i / steps) for i in range(steps + 1)]
    return np.array(interpolated_positions)

def generate_random_pose_and_angle():
    z = random.uniform(0.07, 0.08)
    x = random.uniform(0.5, 0.6)
    y = random.uniform(-0.05, 0.05)
    roll = random.choice([random.uniform(170, 179.99), random.uniform(-179.99, -170)])
    return np.array([x, y, z]), roll

def generate_trajectory(initial_pos, initial_roll, total_points, cycles=10):
    pd_list = [initial_pos]
    rd_list = [initial_roll]
    for _ in range(cycles):
        # Phase 1: Move to random pose and angle
        des_pos, des_roll = generate_random_pose_and_angle()
        pd_list.append(des_pos)
        rd_list.append(des_roll)
        
        # Phase 2: Return to initial pose and angle
        pd_list.append(initial_pos)
        rd_list.append(initial_roll)
    
    interpolated_positions = []
    interpolated_angles = []
    for i in range(len(pd_list) - 1):
        # Interpolate positions
        pd = interpolate_positions(pd_list[i], pd_list[i + 1], total_points)
        interpolated_positions.append(pd[:-1])  # Exclude last point to avoid duplication
        
        # Interpolate angles
        rolls = interpolate_angles(rd_list[i], rd_list[i + 1], total_points)
        interpolated_angles.extend(rolls[:-1])
    
    # Add final point manually to complete cycle
    interpolated_positions.append(pd_list[-1][np.newaxis, :])
    interpolated_angles.append(rd_list[-1])
    
    # Concatenate all phases
    pd_final = np.vstack(interpolated_positions)
    rolls_final = np.array(interpolated_angles)
    
    return pd_final, rolls_final

def main():
    initial_pos = np.array([0.521, 0, 0.2])
    initial_roll = 179.99
    total_points = 100  # Adjusted for demonstration; increase as needed
    
    pd_final, rolls_final = generate_trajectory(initial_pos, initial_roll, total_points)
    print("Final positions shape:", pd_final.shape)
    print("Final angles length:", len(rolls_final))

    # Here you can follow up with rotation calculations, gripper setup, and file saving as before

if __name__ == "__main__":
    main()
