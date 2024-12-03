import jax
import jax.numpy as jnp
import json
import logging
from jax.scipy.spatial.transform import Rotation as R

class MotionProcessor:
    def __init__(self):
        self._LOOP_MODE_KEY = "LoopMode"
        self._FRAME_DURATION_KEY = "FrameDuration"
        self._ENABLE_CYCLE_OFFSET_POSITION_KEY = "EnableCycleOffsetPosition"
        self._ENABLE_CYCLE_OFFSET_ROTATION_KEY = "EnableCycleOffsetRotation"
        self._FRAMES_KEY = "Frames"
        self.POS_SIZE = 3  # Assumed position size
        self.ROT_SIZE = 4  # Assumed quaternion rotation size

    def load(self, motion_file):
        """Load motion data from file using JAX."""
        logging.info("Loading motion from: {:s}".format(motion_file))
        with open(motion_file, "r") as f:
            motion_json = json.load(f)

            self._loop_mode = motion_json[self._LOOP_MODE_KEY]
            self._frame_duration = float(motion_json[self._FRAME_DURATION_KEY])

            self._enable_cycle_offset_pos = bool(
                motion_json.get(self._ENABLE_CYCLE_OFFSET_POSITION_KEY, False)
            )
            self._enable_cycle_offset_rot = bool(
                motion_json.get(self._ENABLE_CYCLE_OFFSET_ROTATION_KEY, False)
            )

            frames = jnp.array(motion_json[self._FRAMES_KEY])
            self._frames = self._postprocess_frames(frames)

            assert self._frames.shape[0] > 0, "Must have at least 1 frame."
            assert self._frames.shape[1] > self.POS_SIZE + self.ROT_SIZE, (
                "Frames have too few degrees of freedom."
            )
            assert self._frame_duration > 0, "Frame duration must be positive."
            
            logging.info("Loaded motion from {:s}.".format(motion_file))
        self.num_frames = self._frames.shape[0]
        self.cycle_delta_pos = self._calc_cycle_delta_pos()
        self.cycle_delta_rot = self._calc_cycle_delta_rot()

    def _postprocess_frames(self, frames):
        """Postprocess frames using JAX."""
        num_frames = frames.shape[0]
        if num_frames > 0:
            first_frame = frames[0]
            pos_start = self.get_frame_root_pos(first_frame)

            def process_frame(frame, pos_start):
                root_pos = self.get_frame_root_pos(frame)
                root_pos = root_pos.at[:2].add(-pos_start[:2])  # Adjust X, Y

                root_rot = self.get_frame_root_rot(frame)
                root_rot = self.normalize_and_standardize_quaternion(root_rot)

                # Update frame with modified values
                frame = self.set_frame_root_pos(root_pos, frame)
                frame = self.set_frame_root_rot(root_rot, frame)
                return frame

            # Vectorized processing of frames
            frames = jax.vmap(lambda frame: process_frame(frame, pos_start))(frames)

        return frames


    def get_frame(self, step_idx):
        frame_idx = step_idx % self.num_frames
        cycle_count = step_idx // self.num_frames
        cycle_offset_pos = self._calc_cycle_offset_pos(cycle_count)
        cycle_offset_rot = self._calc_cycle_offset_rot(cycle_count)

        frame = self._frames[frame_idx]
        cycle_offset_pos = self._calc_cycle_offset_pos(cycle_count)
        cycle_offset_rot = self._calc_cycle_offset_rot(cycle_count)

        # Get the root position and rotation of the frame
        root_pos = self.get_frame_root_pos(frame)
        root_rot = self.get_frame_root_rot(frame)

        root_pos = self.quaternion_rotate_point(root_pos, cycle_offset_rot) + cycle_offset_pos
        root_rot = self.quaternion_multiply(root_rot, cycle_offset_rot)

        # exchnage x and y in root_pos
        # temp = root_pos[0]  
        # root_pos = root_pos.at[0].set(root_pos[1])
        # root_pos = root_pos.at[1].set(temp)
        frame = self.set_frame_root_pos(root_pos, frame)
        frame = self.set_frame_root_rot(root_rot, frame)
        return frame

    def _calc_cycle_delta_pos(self):
        """
        Calculate the net change in the root position after a cycle.
        Returns:
          Net translation of the root position along the horizontal plane.
        """
        first_frame = self._frames[0]
        last_frame = self._frames[-1]

        # Extract root positions
        pos_start = self.get_frame_root_pos(first_frame)
        pos_end = self.get_frame_root_pos(last_frame)

        # Calculate delta position
        cycle_delta_pos = pos_end - pos_start
        cycle_delta_pos = cycle_delta_pos.at[2].set(0)  # Zero out vertical translation

        return cycle_delta_pos

    def _calc_cycle_delta_rot(self):
        """
        Calculate the net change in the root rotation after a cycle.
        Returns:
          Net rotation of the root rotation.
        """
        first_frame = self._frames[0]
        last_frame = self._frames[-1]

        rot_start = self.get_frame_root_rot(first_frame)
        rot_end = self.get_frame_root_rot(last_frame)

        # Compute inverse of start rotation
        inv_rot_start = R.from_quat(rot_start).inv().as_quat()

        # Compute delta rotation
        drot = R.from_quat(rot_end) * R.from_quat(inv_rot_start)
        delta_quaternion = drot.as_quat()
        cycle_delta_heading = self.calc_heading(delta_quaternion)
        return cycle_delta_heading

    def _calc_cycle_offset_pos(self, num_cycles):
        """
        Calculate the net change in root position after a given number of cycles.
        Args:
        num_cycles: Number of cycles since the start of the motion.
        Returns:
        Net translation of the root position.
        """
        # Case 1: No cycle offset position enabled
        zero_offset = jnp.zeros(3)

        # Case 2: Cycle offset position enabled without rotation
        no_rotation_offset = num_cycles * self.cycle_delta_pos

        # Case 3: Cycle offset position enabled with rotation
        def compute_rotational_offset(num_cycles):
            def update_offset(cycle, acc_offset):
                curr_heading = cycle * self.cycle_delta_rot
                rot = self.quaternion_about_axis(curr_heading, jnp.array([0, 0, 1]))
                curr_offset = self.quaternion_rotate_point(self.cycle_delta_pos, rot)
                return acc_offset + curr_offset

            # cycle_range = jnp.arange(num_cycles)
            return jax.lax.fori_loop(0, num_cycles, update_offset, jnp.zeros(3))

        rotational_offset = compute_rotational_offset(num_cycles)
        return no_rotation_offset
        # Combine cases using jax.lax.cond
        return jax.lax.cond(
            self._enable_cycle_offset_pos,
            lambda _: jax.lax.cond(
                self._enable_cycle_offset_rot,
                lambda _: rotational_offset,
                lambda _: no_rotation_offset,
                None,
            ),
            lambda _: zero_offset,
            None,
        )

    def _calc_cycle_offset_rot(self, num_cycles):
        """
        Calculate the net change in root rotation after a given number of cycles.
        Args:
          num_cycles: Number of cycles since the start of the motion.
        Returns:
          Net rotation of the root orientation as a quaternion.
        """
        # Default rotation (identity quaternion) if rotation offsets are disabled
        identity_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

        def compute_rotation(_):
            # Compute heading offset
            heading_offset = num_cycles * self.cycle_delta_rot
            # Create quaternion from heading
            return self.quaternion_from_euler(0.0, 0.0, heading_offset)

        # Use JAX's lax.cond for conditional control flow
        cycle_offset_rot = jax.lax.cond(
            self._enable_cycle_offset_rot,
            compute_rotation,
            lambda _: identity_quaternion,
            operand=None,
        )
        # cycle_offset_rot = compute_rotation(None)
        return cycle_offset_rot


    @staticmethod
    def quaternion_multiply(quat1, quat2):
        rot1 = R.from_quat(quat1)
        rot2 = R.from_quat(quat2)

        # Perform quaternion multiplication
        result = rot1 * rot2

        # Return the resulting quaternion
        return result.as_quat()

    @staticmethod
    def quaternion_from_euler(roll, pitch, yaw):
        """
        Create a quaternion from Euler angles.
        Args:
          roll: Rotation around the x-axis in radians.
          pitch: Rotation around the y-axis in radians.
          yaw: Rotation around the z-axis in radians.
        Returns:
          Quaternion as a 4D vector [w, x, y, z].
        """
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        return rotation.as_quat()

    @staticmethod
    def calc_heading(delta_quaternion):

        rotation = R.from_quat(delta_quaternion)
        euler_angles = rotation.as_euler('xyz', degrees=False)  # Get angles in radians
        return euler_angles[2] 

    @staticmethod
    def quaternion_about_axis(angle, axis):
        """
        Create a quaternion for rotation about an axis.
        Args:
          angle: Rotation angle in radians.
          axis: Axis of rotation as a 3D vector.
        Returns:
          Quaternion as a 4D vector [w, x, y, z].
        """
        axis = jnp.array(axis) / jnp.linalg.norm(axis)
        half_angle = angle / 2
        w = jnp.cos(half_angle)
        xyz = jnp.sin(half_angle) * axis
        return jnp.concatenate([jnp.array([w]), xyz])

    @staticmethod
    def quaternion_rotate_point(point, quaternion):
        """
        Rotate a point using a quaternion.
        Args:
          point: A 3D point.
          quaternion: A 4D quaternion [w, x, y, z].
        Returns:
          Rotated 3D point.
        """
        rotation = R.from_quat(quaternion)
        return rotation.apply(point)


    @staticmethod
    def normalize_and_standardize_quaternion(q):
        """Normalize and standardize quaternion."""
        q = q / jnp.linalg.norm(q)
        # print(q)
        # if q[0] < 0:
        #     q = -q  # Standardize quaternion
        return q

    @staticmethod
    def get_frame_root_pos(frame):
        """Extract root position from a frame."""
        return frame[:3]

    @staticmethod
    def set_frame_root_pos(pos, frame):
        """Set root position in a frame."""
        frame = frame.at[:3].set(pos)
        return frame

    @staticmethod
    def get_frame_root_rot(frame):
        """Extract root rotation (quaternion) from a frame."""
        return frame[3:7]

    @staticmethod
    def set_frame_root_rot(rot, frame):
        """Set root rotation in a frame."""
        frame = frame.at[3:7].set(rot)
        return frame
