# Legislative Harness Class for VLA Bench
import copy
import numpy as np

class LegislativeHarness:
	
    def __init__(self, action_hist_len=5, n_action_steps=10):
        self.action_history = []
        self.action_hist_len = action_hist_len
        self.n_action_steps = n_action_steps

    def update_history(self, action):
        self.action_history.append(action)
        self.action_history = self.action_history[-self.action_hist_len:]

    def is_illegal_action(
            self,
            physics,            # dm_control Physics object
            task,               # LM4ManipBaseTask (has .entities and .robot)
            law: dict | None,
            action_traj: np.ndarray,
        ) -> bool:
        """
        Return True if executing action_traj would violate the given law.
 
        Parameters
        ----------
        physics      : dm_control Physics — passed through from the env step.
        task         : VLABench task object — provides task.entities and task.robot.
        law          : A single law dict from LegislativeModule, or None.
        action_traj  : numpy array (T, action_dim).
        """
        if law is None:
            return False, ''
 
        obj_name = law["Symbolic"]["Object"]
        if obj_name is None:
            return False, ''
 
        illegal_pred = law["Symbolic"]["Predicate"]
 
        # Does the illegal object exist in this scene?
        if obj_name not in task.entities:
            return False, ''
 
        illegal_entity = task.entities[obj_name]
 
        # Grasp detection: entity already in contact with robot gripper
        illegally_grasped = illegal_entity.is_grasped(physics, task.robot)

        if illegally_grasped: 
            print(f"[Debug] Illegal Grasp Detected")

        # Gripper intent: closing now, or currently closed
        closing = self.is_closing(action_traj)
        if closing:
            print("[Debug] Closing Detected")

        is_closed = self.is_closed(physics, task.robot)
        if is_closed:
            print("[Debug] Closed Detected")

        return illegally_grasped and (closing or is_closed), obj_name

    def is_closed(self, physics, robot) -> bool:
        """
        Return True when the gripper is in a closed / near-closed state.
 
        VLABench robots expose their gripper as robot.gripper, which is a
        dm_control MJCFEntity.  We bind its joints and read qpos.
        A mean joint position below 0.1 rad is treated as closed (same
        threshold as the Libero version).
        """
        try:
            gripper_joints = robot.gripper.joints
            positions = physics.bind(gripper_joints).qpos
            return float(np.mean(positions)) < 0.02
        except Exception:
            return False

    def is_closing(self, action_traj) -> bool:
        if action_traj.ndim == 1:
            return float(np.mean(action_traj[-2:])) < 0.02
        execute_window = action_traj[:self.n_action_steps]
        gripper_cmds = execute_window[:, -2:]
        return float(np.mean(gripper_cmds[-5:])) < 0.02

    def action_filter(
            self,
            physics,
            task,
            law: dict | None,
            action_traj: np.ndarray,
        ) -> np.ndarray:
        """
        If the trajectory would violate the law, replace it with a safe
        trajectory: freeze the arm (zero Cartesian/joint deltas) and open
        the gripper.  Otherwise return action_traj unchanged.
 
        Returns a numpy array of the same shape as action_traj.
        """
        is_illegal, illegal_entity = self.is_illegal_action(physics, task, law, action_traj)
        if is_illegal:
            legal_traj = action_traj.copy()
            current_qpos = np.array(task.robot.get_qpos(physics)).reshape(-1)
            if legal_traj.ndim == 1:
                legal_traj[:-2] = current_qpos
                legal_traj[-2:] = 0.04  # open gripper
            else:
                legal_traj[:, :-2] = current_qpos
                legal_traj[:, -2:] = 0.04
            return legal_traj, illegal_entity

        return action_traj, ''
 
