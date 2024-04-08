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

def generate_random_target(only_xy=False, current_pos=None):
    """
    Generates a random target position and roll angle within specified ranges.
    If only_xy is True, generates a new position with random x and y, keeping z and roll of current_pos.
    """
    if only_xy and current_pos is not None:
        z, roll = current_pos[2], current_pos[3]
    else:
        z = random.uniform(0.07, 0.078)
        roll = random.choice([-179.99, -175, 175, 179.99])
    
    x = random.uniform(0.52, 0.58)
    y = random.uniform(-0.03, 0.03)
    return np.array([x, y, z]), roll

def setup_gripper_rotation(total_points, time, start_time_gripper_turn, end_time_gripper_turn):
    gripper_rot_speed = 2 * np.pi / (end_time_gripper_turn - start_time_gripper_turn)
    gripper_angles = np.zeros(total_points)
    for i in range(total_points):
        if start_time_gripper_turn <= time[i] <= end_time_gripper_turn:
            elapsed_time = time[i] - start_time_gripper_turn
            gripper_angles[i] = gripper_rot_speed * elapsed_time
    return np.array([R.from_euler('z', angle).as_matrix() for angle in gripper_angles])

def generate_trajectory(N, total_time, total_points):
    initial_pos = np.array([0.521, 0, 0.2])
    initial_roll = 179.99

    concatenated_pd = []
    concatenated_Rd = []

    for cycle in range(N):
        des_pos, des_roll = generate_random_target()

        phase_points = total_points // (3 * N)
        hold_points = phase_points

        pd_phase1 = np.array([initial_pos + (des_pos - initial_pos) * (i / phase_points) for i in range(phase_points)])
        
        # Decide between holding and moving to another random location
        choice = random.choice(['hold', 'move'])
        if choice == 'hold':
            pd_mid_phase = np.repeat(des_pos[np.newaxis, :], hold_points, axis=0)
            mid_phase_roll = des_roll
        else:
            new_des_pos, _ = generate_random_target(only_xy=True, current_pos=np.append(des_pos, des_roll))
            pd_mid_phase = np.array([des_pos + (new_des_pos - des_pos) * (i / hold_points) for i in range(hold_points)])
            mid_phase_roll = des_roll  # Keep roll constant for simplicity

        pd_phase2 = np.array([pd_mid_phase[-1] + (initial_pos - pd_mid_phase[-1]) * (i / phase_points) for i in range(phase_points)])
        
        pd_cycle = np.vstack((pd_phase1, pd_mid_phase, pd_phase2))
        
        rolls_phase1 = interpolate_angles(initial_roll, des_roll, phase_points)
        rolls_mid_phase = [mid_phase_roll] * hold_points
        rolls_phase2 = interpolate_angles(mid_phase_roll, initial_roll, phase_points)
        
        rolls = np.concatenate((rolls_phase1, rolls_mid_phase, rolls_phase2))
        pitchs = [0] * len(rolls)
        yaws = [0] * len(rolls)
        
        rots = R.from_euler('xyz', np.column_stack((rolls, pitchs, yaws)), degrees=True)
        
        Rd_cycle = np.array([rot.as_matrix() for rot in rots])

        
        
        concatenated_pd.append(pd_cycle)
        concatenated_Rd.append(Rd_cycle)
    
    final_pd = np.vstack(concatenated_pd)
    final_Rd = np.vstack(concatenated_Rd).reshape(-1, 3, 3)
    final_Rd_gripper = np.vstack([np.eye(3)] * final_pd.shape[0])
    print("here")
    np.savetxt('pd.txt', final_pd, fmt='%.6f')
    np.savetxt('Rd.txt', final_Rd.reshape(final_Rd.shape[0]*3, -1), fmt='%.6f')
    np.savetxt('Rd_gripper.txt', final_Rd_gripper.reshape(final_Rd.shape[0]*3, -1), fmt='%.6f')

    return final_pd.shape, final_Rd.shape

def main():
    initial_pos = np.array([0.521, 0, 0.2])
    initial_roll = 179.99
    total_points = 100  # Adjusted for demonstration; increase as needed
    
    N = 25
    total_time = 10
    total_points = 10000

    pd_shape, Rd_shape = generate_trajectory(N, total_time, total_points)
    print("Shapes: pd {}, Rd {}".format(pd_shape, Rd_shape))

    # Here you can follow up with rotation calculations, gripper setup, and file saving as before

if __name__ == "__main__":
    main()
