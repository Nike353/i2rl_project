import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def read_and_plot_yaw(json_file):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract frames
    frames = np.array(data['Frames'])
    
    # Extract orientation (4th to 7th columns) as quaternions
    quaternions = frames[:, 3:7]
    
    # normalize the quaternions
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1)[:, np.newaxis]
    
    
    # Convert quaternions to roll, pitch, yaw
    rotations = R.from_quat(quaternions)
    rpy_angles = rotations.as_euler('xyz', degrees=False)  # Convert to degrees
    yaw_angles = rpy_angles[:, 2]  # Extract yaw (z-axis rotation)
    
    # Plot yaw angle
    plt.figure(figsize=(10, 6))
    plt.plot(yaw_angles, label="Yaw Angle")
    plt.xlabel("Frame Index")
    plt.ylabel("Yaw Angle (degrees)")
    plt.title("Yaw Angle Over Frames")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
json_file = '/home/guanqihe/nikhil/i2rl_project/assets/go2_right turn0.txt'  # Replace with the path to your JSON file
read_and_plot_yaw(json_file)
