import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
loaded_data = np.load("/home/guanqihe/nikhil/i2rl_project/unitree_go2_trajectory/20241202-224826_states.npy")

# plot x, y,z position
fig, axs = plt.subplots(3, 1)
axs[0].plot(loaded_data[:, 0], loaded_data[:, 1])
axs[1].plot(loaded_data[:, 0], loaded_data[:, 2])
axs[2].plot(loaded_data[:, 0], loaded_data[:, 3])
plt.show()

# read all the quat columns and convert to euler angles and plot yaw
quat_data = loaded_data[:, 4:8]
euler_data = R.from_quat(quat_data).as_euler('xyz', degrees=True)
plt.plot(loaded_data[:, 0], euler_data[:, 2])
plt.show()