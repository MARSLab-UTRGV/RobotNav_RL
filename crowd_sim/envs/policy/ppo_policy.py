import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import argparse
from torch.optim import Adam

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class PPONetwork(nn.Module):
    """Neural network for the PPO policy and value function"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(hidden_dim, output_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(output_dim))
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """Forward pass through network"""
        features = self.feature_extractor(x)
        
        # Get action mean and std
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_logstd)
        
        # Get value
        value = self.value(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy distribution"""
        # Get mean and std
        action_mean, action_std, _ = self.forward(state)
        
        # If deterministic, return mean
        if deterministic:
            # Return mean and a dummy log prob (not used during evaluation)
            return action_mean, torch.zeros(action_mean.shape[0], device=action_mean.device)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Sample action
        action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, states, actions):
        """Evaluate actions given states"""
        # Get mean, std, and value
        action_mean, action_std, values = self.forward(states)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Get log probability
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Get entropy
        entropy = dist.entropy().sum(dim=-1).mean()
        
        return log_probs, values.squeeze(), entropy


class PPOPolicy(Policy):
    """
    PPO policy for robot navigation
    """
    
    def __init__(self):
        """
        Initialize PPO policy
        """
        super().__init__()
        self.name = 'PPOPolicy'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.with_om = False
        
        # PPO specific parameters
        self.max_action = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Policy initialized with device: {self.device}")
        self.model = None
        self.state_dim = None
        self.action_dim = 2  # vx, vy
        self.hidden_dim = 128
        self.model_path = None
        
    def configure(self, config):
        """
        Configure the policy using parameters from config
        """
        if config.has_section('ppo'):
            if config.has_option('ppo', 'model_path'):
                self.model_path = config.get('ppo', 'model_path')
            
            if config.has_option('ppo', 'hidden_dim'):
                self.hidden_dim = config.getint('ppo', 'hidden_dim')
        
        self.with_om = config.getboolean('robot', 'with_om', fallback=False)
                
        return
    
    def set_phase(self, phase):
        """
        Set the phase of training or testing
        """
        return
    
    def set_device(self, device):
        """
        Set the device for the model
        """
        self.device = device
        print(f"Setting PPO Policy device to: {self.device}")
        if self.model is not None:
            self.model = self.model.to(device)
    
    def build_model(self):
        """
        Build the neural network model
        """
        # State-dimension depends on observation
        if self.state_dim is None:
            raise ValueError("state_dim is not set, cannot build model")
            
        # Create model
        print(f"Building PPO model with input_dim={self.state_dim}, action_dim={self.action_dim}, hidden_dim={self.hidden_dim} on device {self.device}")
        self.model = PPONetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.model = self.model.to(self.device)
        
        # Load model if path is provided
        if self.model_path:
            print(f"Loading model from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
    
    def get_state_tensor(self, state):
        """
        Convert state to tensor for model input
        
        Args:
            state: JointState object with robot state and observation
            
        Returns:
            torch.Tensor: state tensor
        """
        robot_state = state.self_state
        
        # Robot state features: px, py, vx, vy, radius, gx, gy, v_pref
        robot_features = torch.tensor([
            robot_state.px, robot_state.py,
            robot_state.vx, robot_state.vy,
            robot_state.radius,
            robot_state.gx, robot_state.gy,
            robot_state.v_pref
        ], dtype=torch.float32, device=self.device)
        
        # Features for each human
        human_features = []
        
        for human in state.human_states:
            # Human features: px, py, vx, vy, radius
            features = [
                human.px - robot_state.px, 
                human.py - robot_state.py,
                human.vx, 
                human.vy,
                human.radius + robot_state.radius
            ]
            
            # Add to human features list
            human_features.extend(features)
        
        # Convert to tensor and combine with robot features
        human_tensor = torch.tensor(human_features, dtype=torch.float32, device=self.device)
        state_tensor = torch.cat([robot_features, human_tensor])
        
        # Update state_dim if not set
        if self.state_dim is None:
            self.state_dim = state_tensor.shape[0]
            
        return state_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, state):
        """
        Predict action based on state
        
        Args:
            state: JointState object with robot state and observation
            
        Returns:
            ActionXY: action with vx, vy components
        """
        # Try to initialize the model if not already initialized
        self._ensure_model_initialized(state)
        
        # Now the model should be initialized
        if self.model is None:
            raise RuntimeError("Failed to initialize the model. This should not happen.")
        
        # Convert state to tensor
        state_tensor = self.get_state_tensor(state)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get action
        with torch.no_grad():
            # During testing, use deterministic action (mean)
            action, _ = self.model.get_action(state_tensor, deterministic=True)
            
        # Convert action to numpy
        action = action.cpu().numpy().squeeze()
        
        # Clip action to max_action
        action = np.clip(action, -self.max_action, self.max_action)
        
        # Return ActionXY
        self.last_state = state
        return ActionXY(action[0], action[1])
    
    def _ensure_model_initialized(self, state):
        """
        Ensure the model is initialized
        
        Args:
            state: JointState object to use for initialization if needed
        """
        if self.model is None:
            # Get state tensor to determine state_dim
            state_tensor = self.get_state_tensor(state)
            if self.state_dim is None:
                self.state_dim = state_tensor.shape[1]
            # Build model now that we have the state dimension
            self.build_model()
            print(f"Model initialized with state_dim={self.state_dim}")
            return True
        return False
    
    def save_model(self, path):
        """
        Save model to path
        """
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """
        Load model from path
        """
        self.model_path = path
        if self.model is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
        # If model is not built yet, it will be loaded when built 


# Function to test a trained model
def test_model(env, model, num_episodes=10):
    """Test a trained model in the environment and render the results
    
    Args:
        env: Environment to test in
        model: Trained model to test
        num_episodes: Number of episodes to run
    """
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            # Format observation for model
            robot_state = obs.self_state
            
            # Robot features
            robot_features = torch.tensor([
                robot_state.px, robot_state.py,
                robot_state.vx, robot_state.vy,
                robot_state.radius,
                robot_state.gx, robot_state.gy,
                robot_state.v_pref
            ], dtype=torch.float32, device=device).unsqueeze(0)
            
            # Human features
            human_features_list = []
            visible_masks = []
            
            for human in obs.human_states:
                # Human features
                features = [
                    human.px - robot_state.px,
                    human.py - robot_state.py,
                    human.vx,
                    human.vy,
                    human.radius + robot_state.radius
                ]
                human_features_list.append(features)
                
                # Visibility mask
                visible_masks.append(1.0 if getattr(human, 'visible', True) else 0.0)
            
            # Pad to fixed size if necessary
            max_humans = 5  # Assuming 5 humans max
            while len(human_features_list) < max_humans:
                human_features_list.append([0, 0, 0, 0, 0])
                visible_masks.append(0.0)
            
            # Convert to tensors
            human_features = torch.tensor(human_features_list, dtype=torch.float32, device=device).unsqueeze(0)
            visible_masks = torch.tensor(visible_masks, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Create input dict
            obs_dict = {
                'robot_node': robot_features,
                'spatial_edges': human_features,
                'visible_masks': visible_masks
            }
            
            # Get action from model
            with torch.no_grad():
                _, action_logits, _ = model(obs_dict, infer=True)
                
            # Convert to actual action
            action = action_logits.cpu().numpy().squeeze()
            
            # Take step in environment
            action_xy = ActionXY(action[0], action[1])
            obs, reward, terminated, truncated, info = env.step(action_xy)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            # Render environment
            env.render()
            
            # Print step info
            print(f"Step {step_count}: Action [{action[0]:.2f}, {action[1]:.2f}], Reward: {reward:.2f}")
            
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps: {step_count}") 