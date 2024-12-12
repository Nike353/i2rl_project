import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
loaded_data = np.load("/home/guanqihe/nikhil/i2rl_project/assets/unitree_go2_trajectory/20241203-080219_states.npy")

# plot x, y,z position
fig, axs = plt.subplots(3, 1)
axs[0].plot(loaded_data[:, 0], loaded_data[:, 1],label="x")
axs[0].plot(loaded_data[:, 0], loaded_data[:, -4],label="x_ref")
axs[1].plot(loaded_data[:, 0], loaded_data[:, 2],label="y")
axs[1].plot(loaded_data[:, 0], loaded_data[:, -3],label="y_ref")
axs[2].plot(loaded_data[:, 0], loaded_data[:, 3],label="z")
axs[2].plot(loaded_data[:, 0], loaded_data[:, -2],label="z_ref")
axs[0].legend()
axs[1].legend()
axs[2].legend()
rmse_x = np.sqrt(np.mean((loaded_data[:, 1] - loaded_data[:, -4])**2))
rmse_y = np.sqrt(np.mean((loaded_data[:, 2] - loaded_data[:, -3])**2))
rmse_z = np.sqrt(np.mean((loaded_data[:, 3] - loaded_data[:, -2])**2))

print(f"RMSE of x: {rmse_x}, RMSE of y: {rmse_y}, RMSE of z: {rmse_z}")
print(f"RMSE of position: {np.sqrt((rmse_x**2 + rmse_y**2 + rmse_z**2))/3}")
#rmse of the yaw tracking error
quat_data = loaded_data[:, 4:8]
euler_data = R.from_quat(quat_data).as_euler('zyx', degrees=False)
yaw_error = loaded_data[100:200, -1] - (-euler_data[100:200, 2]-3.14)
rmse_yaw = np.sqrt(np.mean(yaw_error**2))
print(f"Mean of the yaw tracking error: {rmse_yaw}")
fig.suptitle(f"Position Tracking Performance for Trotting with RMSE of position: {np.sqrt((rmse_x**2 + rmse_y**2 + rmse_z**2)/3):.2f}")
plt.show()

# read all the quat columns and convert to euler angles and plot yaw

plt.plot(loaded_data[:, 0], loaded_data[:, -1])
plt.plot(loaded_data[:, 0], -euler_data[:, 2]-3.14)
plt.show()

##find rmse of the position tracking as entire position trajectory
