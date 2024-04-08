import numpy as np
from scipy.spatial.transform import Rotation as R

def interpolate_angles(start_angle, end_angle, steps):
    """
    Interpolates from start_angle to end_angle (in degrees) over 'steps' increments,
    handling wrap-around at the -180/180 boundary.
    """
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

def interpolate_positions(initial_pos, des_pos, total_points, start_idx_gripper_turn):
    pd = np.array([initial_pos + (des_pos - initial_pos) * (i / start_idx_gripper_turn) for i in range(start_idx_gripper_turn)])
    pd = np.vstack((pd, np.repeat(des_pos[np.newaxis, :], total_points - start_idx_gripper_turn, axis=0)))
    return pd

def calculate_rotations(initial_roll, des_roll, start_idx_gripper_turn, total_points):
    rolls = interpolate_angles(initial_roll, des_roll, start_idx_gripper_turn)
    pitchs = [0] * start_idx_gripper_turn
    yaws = [0] * start_idx_gripper_turn
    rots_until_start = R.from_euler('xyz', np.column_stack((rolls, pitchs, yaws)), degrees=True)
    final_quat = rots_until_start.as_quat()[-1]
    final_rots = np.tile(final_quat, (total_points - start_idx_gripper_turn, 1))
    rots = R.from_quat(np.vstack((rots_until_start.as_quat(), final_rots)))
    return np.array([rot.as_matrix() for rot in rots])

def setup_gripper_rotation(total_points, time, start_time_gripper_turn, end_time_gripper_turn):
    gripper_rot_speed = 2 * np.pi / (end_time_gripper_turn - start_time_gripper_turn)
    gripper_angles = np.zeros(total_points)
    for i in range(total_points):
        if start_time_gripper_turn <= time[i] <= end_time_gripper_turn:
            elapsed_time = time[i] - start_time_gripper_turn
            gripper_angles[i] = gripper_rot_speed * elapsed_time
    return np.array([R.from_euler('z', angle).as_matrix() for angle in gripper_angles])

def main():
    # Inputs
    des_pos = np.array([0.521, 0, 0.01])
    des_roll = 179.99
    total_points = 1000
    total_time = 5
    start_time_gripper_turn = 0.5 * total_time
    end_time_gripper_turn = total_time
    initial_pos = np.array([0.521, 0, 0.2])
    initial_roll = 179.99

    time = np.linspace(0, total_time, total_points)
    start_idx_gripper_turn = int(total_points * (start_time_gripper_turn / total_time))

    pd = interpolate_positions(initial_pos, des_pos, total_points, start_idx_gripper_turn)
    Rd = calculate_rotations(initial_roll, des_roll, start_idx_gripper_turn, total_points)
    Rd_gripper = setup_gripper_rotation(total_points, time, start_time_gripper_turn, end_time_gripper_turn)

    np.savetxt('pd.txt', pd.reshape(total_points, -1), fmt='%.6f')
    np.savetxt('Rd.txt', Rd.reshape(total_points*3, -1), fmt='%.6f')
    np.savetxt('Rd_gripper.txt', Rd_gripper.reshape(total_points*3, -1), fmt='%.6f')

    print("Shapes: pd {}, Rd {}, Rd_gripper {}".format(pd.shape, Rd.shape, Rd_gripper.shape))

if __name__ == "__main__":
    main()
