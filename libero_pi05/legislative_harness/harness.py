# Legislative Harness Class
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

    def is_illegal_action(self, env, law, action_traj:list) -> bool:
        if law is None:
            return False
        obj_name = law["PDDL"]["Object"]
        if obj_name is None:
            return False

        illegal_pred = law["PDDL"]["Predicate"]

        closing = self.is_closing(action_traj)
        e = env.envs[0]._env.env
        is_closed = self.is_closed(e)

        # For now just defining illegal action = grasp
        if illegal_pred == "Grasp":
            illegal_obj = e.objects_dict[obj_name]
            contact = e.check_contact(
                e.robots[0].gripper,
                illegal_obj.contact_geoms
            )
            return (contact and (closing or is_closed))

        return False 

    def is_closed(self, e) -> bool:
        qpos = e.sim.data.qpos
        gripper_joints = e.robots[0].gripper.joints
        gripper_positions = [qpos[e.sim.model.joint_name2id(j)] for j in gripper_joints]
        return np.mean(gripper_positions) < 0.1

    def is_closing(self, action_traj) -> bool:
        execute_window = action_traj[:self.n_action_steps]  # (10, 7)

        gripper_cmds = execute_window[:, -1]  # last dim is gripper
        closing = np.mean(gripper_cmds[-5:]) < 0  # last 5 trending closed
        return closing

    def action_filter(self, env, law, action_traj:list) -> list:
        is_illegal_action = self.is_illegal_action(env, law, action_traj)

        if is_illegal_action:
            print("[debug] Illegal Action")
            legal_traj = action_traj.copy()
            legal_traj[:, :-1] = 0   # freeze arm
            legal_traj[:,-1] = 1
            
            return legal_traj 

        return action_traj
