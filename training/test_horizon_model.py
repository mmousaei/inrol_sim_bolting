import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib
import rosbag
from geometry_msgs.msg import WrenchStamped, PoseStamped
from sensor_msgs.msg import JointState
from network_structures.RNN_horizon import DynamicsModelRNN

def read_specific_bagfile(bag_file_path):
    """
    Reads data from a specific .bag file for testing.
    """
    f_ee = []
    states = []
    actions = []

    bag = rosbag.Bag(bag_file_path)
    for topic, msg, t in bag.read_messages(topics=['/contact_wrench', '/franka_joint_angle', '/obj_pose']):
        if topic == '/contact_wrench':
            f_ee.append([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        elif topic == '/franka_joint_angle':
            actions.append(list(msg.joint_angle))
        elif topic == '/obj_pose':
            states.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                           msg.pose.orientation.x, msg.pose.orientation.y,
                           msg.pose.orientation.z, msg.pose.orientation.w])
    bag.close()

    return np.array(states), np.array(actions), np.array(f_ee)

def prepare_test_sequences(states, actions, f_ee, sequence_length=50):
    X = []
    for i in range(len(states) - sequence_length):
        state_sequence = states[i]
        f_ee_sequence = f_ee[i]
        action_sequence = actions[i:i+sequence_length].flatten()
        X.append(np.concatenate([state_sequence, f_ee_sequence, action_sequence]))
    return np.array(X)

def plot_forces(true_forces, predicted_forces):
    time_steps = range(true_forces.shape[0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    force_labels = ['Force X', 'Force Y', 'Force Z']
    for i in range(3):
        axs[i].plot(time_steps, true_forces[:, i], label='Ground Truth', marker='o')
        axs[i].plot(time_steps, predicted_forces[:, i], label='Predicted', linestyle='--')
        axs[i].set_ylabel(force_labels[i])
        axs[i].legend()
    axs[2].set_xlabel('Time Step')
    plt.show()

def test_model(bag_file_path, model_path, scaler_path):
    
    # Load the trained model and scaler
    model = DynamicsModelRNN(input_dim=360 , hidden_dim=128, output_dim=150)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler = joblib.load(scaler_path)

    # Read and prepare test data
    states, actions, f_ee_ground_truth = read_specific_bagfile(bag_file_path)
    X_test = prepare_test_sequences(states, actions, f_ee_ground_truth, sequence_length=50)
    X_test_scaled = scaler.transform(X_test)
    
    # Predict forces
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        print(X_test_tensor.size())
        f_ee_predicted = model(X_test_tensor).numpy()

    # Reshape predictions to match the ground truth shape
    f_ee_predicted_reshaped = f_ee_predicted[0, :].reshape((50, 3))  # Assuming your model predicts flattened sequences
    print("input shape = ", X_test_scaled.shape)
    print("fee pred shape = ", f_ee_predicted.shape)
    print("fee pred re shape = ", f_ee_predicted_reshaped.shape)
    time_steps = range(f_ee_ground_truth.shape[0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    force_labels = ['Force X', 'Force Y', 'Force Z']
    for i in range(3):
        axs[i].plot(time_steps, f_ee_ground_truth[:, i], label='Ground Truth', linewidth=3)
        for j in range(X_test_scaled.shape[0]):
            f_ee_predicted_reshaped = f_ee_predicted[j, :].reshape((50, 3))
            axs[i].plot(time_steps[j:j+50], f_ee_predicted_reshaped[:, i], label='Predicted', linewidth=0.7, alpha=0.5)

        # axs[i].plot(time_steps, predicted_forces[:, i], label='Predicted', linestyle='--')
        axs[i].set_ylabel(force_labels[i])
        # axs[i].legend()
    axs[2].set_xlabel('Time Step')
    plt.show()
    # Plot forces
    # plot_forces(f_ee_ground_truth[50:], f_ee_predicted_reshaped)  # Skip the first `sequence_length` ground truths

if __name__ == "__main__":
    bag_file_path = "data/peginhole_original.bag"  # Update this path
    model_path = "model_epoch_500.pth"  # Update path as needed
    scaler_path = "scaler_contact_force_horizon_full_state.save"  # Update path as needed
    test_model(bag_file_path, model_path, scaler_path)
