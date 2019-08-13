import numpy as np
from env.env import Env
from DeepMimicCore import DeepMimicCore
from env.action_space import ActionSpace
import sys

class DeepMimicEnv(Env):
    def __init__(self, args, enable_draw):
        super().__init__(args, enable_draw)
        
        self._core = DeepMimicCore.cDeepMimicCore(enable_draw)
        ##self._core = DeepMimicCore.cDeepMimicCore(False)
        rand_seed = np.random.randint(np.iinfo(np.int32).max)
        self._core.SeedRand(rand_seed)
        print("##################", sys._getframe().f_code.co_name ,"###################")
        print(args)
        self._core.ParseArgs(args)
        self._core.Init()
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return

    def update(self, timestep):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        self._core.Update(timestep)

    def reset(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        self._core.Reset()

    def get_time(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetTime()

    def get_name(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetName()

    # rendering and UI interface
    def draw(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        self._core.Draw()

    def keyboard(self, key, x, y):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        self._core.Keyboard(key, x, y)

    def mouse_click(self, button, state, x, y):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        self._core.MouseClick(button, state, x, y)

    def mouse_move(self, x, y):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        self._core.MouseMove(x, y)

    def reshape(self, w, h):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        self._core.Reshape(w, h)

    def shutdown(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        self._core.Shutdown()

    def is_done(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.IsDone()

    def set_playback_speed(self, speed):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        print(speed)
        self._core.SetPlaybackSpeed(speed)

    def set_updates_per_sec(self, updates_per_sec):
        print("##################", sys._getframe().f_code.co_name ,updates_per_sec,"###################")
        print(updates_per_sec)
        self._core.SetUpdatesPerSec(updates_per_sec)

    def get_win_width(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetWinWidth()

    def get_win_height(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetWinHeight()

    def get_num_update_substeps(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetNumUpdateSubsteps()

    # rl interface
    def is_rl_scene(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.IsRLScene()

    def get_num_agents(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetNumAgents()

    def need_new_action(self, agent_id):
        print("##################", sys._getframe().f_code.co_name , agent_id,"###################")
        return self._core.NeedNewAction(agent_id)

    def record_state(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.RecordState(agent_id))

    def record_goal(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.RecordGoal(agent_id))

    def get_action_space(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return ActionSpace(self._core.GetActionSpace(agent_id))
    
    def set_action(self, agent_id, action):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        print(agent_id, action.tolist())
        return self._core.SetAction(agent_id, action.tolist())
    
    def get_state_size(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetStateSize(agent_id)

    def get_goal_size(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetGoalSize(agent_id)

    def get_action_size(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetActionSize(agent_id)

    def get_num_actions(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetNumActions(agent_id)

    def build_state_offset(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildStateOffset(agent_id))

    def build_state_scale(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildStateScale(agent_id))
    
    def build_goal_offset(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildGoalOffset(agent_id))

    def build_goal_scale(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildGoalScale(agent_id))
    
    def build_action_offset(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildActionOffset(agent_id))

    def build_action_scale(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildActionScale(agent_id))

    def build_action_bound_min(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildActionBoundMin(agent_id))

    def build_action_bound_max(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildActionBoundMax(agent_id))

    def build_state_norm_groups(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildStateNormGroups(agent_id))

    def build_goal_norm_groups(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return np.array(self._core.BuildGoalNormGroups(agent_id))

    def calc_reward(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.CalcReward(agent_id)

    def get_reward_min(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetRewardMin(agent_id)

    def get_reward_max(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetRewardMax(agent_id)

    def get_reward_fail(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetRewardFail(agent_id)

    def get_reward_succ(self, agent_id):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.GetRewardSucc(agent_id)

    def is_episode_end(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.IsEpisodeEnd()

    def check_terminate(self, agent_id):
        return Env.Terminate(self._core.CheckTerminate(agent_id))

    def check_valid_episode(self):
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return self._core.CheckValidEpisode()

    def log_val(self, agent_id, val):
        self._core.LogVal(agent_id, float(val))
        print("##################", sys._getframe().f_code.co_name ,"###################")
        return

    def set_sample_count(self, count):
        self._core.SetSampleCount(count)
        print("##################", sys._getframe().f_code.co_name , count,"###################")
        return

    def set_mode(self, mode):
        self._core.SetMode(mode.value)
        print("##################", sys._getframe().f_code.co_name , mode,"###################")
        print(mode)
        return
