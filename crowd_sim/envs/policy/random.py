import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot

class RandomPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'RandomPolicy'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.max_speed = 1.0
        self.random_seed = None
    
    def configure(self, config):
        """
        Configure the random policy with parameters from config
        """
        if config.has_section('random'):
            self.max_speed = config.getfloat('random', 'max_speed', fallback=1.0)
            if config.has_option('random', 'random_seed'):
                self.random_seed = config.getint('random', 'random_seed')
                np.random.seed(self.random_seed)
        return

    def set_phase(self, phase):
        """
        Phase doesn't affect random policy
        """
        return

    def predict(self, state):
        """
        Generate random velocity commands within the maximum speed
        
        :param state: current state of the environment
        :return: ActionXY with random vx and vy
        """
        if self.kinematics == 'holonomic':
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0, self.max_speed)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            action = ActionXY(vx, vy)
        else:
            v = np.random.uniform(0, self.max_speed)
            r = np.random.uniform(-np.pi, np.pi)
            action = ActionRot(v, r)
        self.last_state = state
        return action
