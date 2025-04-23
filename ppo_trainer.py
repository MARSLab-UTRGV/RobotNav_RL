import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import configparser
import argparse
from collections import deque
import time
import logging

from crowd_sim.envs import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.ppo_policy import PPOPolicy
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PPO-Trainer")

class PPOTrainer:
    """
    Trainer for PPO policy in robot navigation
    """
    
    def __init__(self, args):
        """
        Initialize PPO trainer
        
        Args:
            args: Command line arguments
        """
        self.args = args
        
        # Training parameters
        self.num_episodes = args.num_episodes
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_param = args.clip_param
        self.value_coef = args.value_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        
        # Create directory for models and logs
        self.save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))
        
        # Create environment
        self.config = self._create_config()
        self.env = CrowdSim()
        self.env.configure(self.config)
        
        # Create robot with PPO policy
        self.policy = PPOPolicy()
        self.robot = Robot(self.config, 'robot')
        self.robot.set_policy(self.policy)
        
        # Set the robot in the environment
        self.env.set_robot(self.robot)
        
        # Initialize optimizer
        self.optimizer = None
        
        # Initialize buffer for experience collection
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
    def _create_config(self):
        """Create a configuration for the simulation."""
        config = configparser.ConfigParser()

        # Environment settings
        config.add_section('env')
        config.set('env', 'time_limit', '25')
        config.set('env', 'time_step', '0.25')
        config.set('env', 'randomize_attributes', 'True')
        config.set('env', 'val_size', '100')
        config.set('env', 'test_size', '100')

        # Reward settings
        config.add_section('reward')
        config.set('reward', 'success_reward', '1')
        config.set('reward', 'collision_penalty', '-1')
        config.set('reward', 'discomfort_dist', '0.2')
        config.set('reward', 'discomfort_penalty_factor', '0.5')

        # Humans settings
        config.add_section('humans')
        config.set('humans', 'policy', 'orca')
        config.set('humans', 'visible', 'True')
        config.set('humans', 'v_pref', '1')
        config.set('humans', 'radius', '0.3')
        config.set('humans', 'sensor', 'coordinates')

        # Simulation settings
        config.add_section('sim')
        config.set('sim', 'train_val_sim', 'circle_crossing')
        config.set('sim', 'test_sim', 'circle_crossing')
        config.set('sim', 'square_width', '6')
        config.set('sim', 'circle_radius', '4')
        config.set('sim', 'human_num', '5')

        # Robot settings
        config.add_section('robot')
        config.set('robot', 'policy', 'ppo')
        config.set('robot', 'radius', '0.3')
        config.set('robot', 'v_pref', '1')
        config.set('robot', 'visible', 'True')
        config.set('robot', 'sensor', 'coordinates')

        # PPO policy settings
        config.add_section('ppo')
        config.set('ppo', 'hidden_dim', '128')
        
        return config
    
    def collect_experience(self, num_steps=1000):
        """
        Collect experience for training
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            dict: Buffer of collected experience
        """
        # Reset buffer
        for key in self.buffer:
            self.buffer[key] = []
            
        # Reset environment
        obs = self.env.reset(phase='train')
        
        # Ensure model is initialized
        if self.policy.model is None:
            state = JointState(self.robot.get_full_state(), obs)
            _ = self.policy.get_state_tensor(state)  # This will trigger model building
            
            # Additional check to ensure model is built
            if self.policy.model is None:
                self.policy._ensure_model_initialized(state)
                if self.policy.model is None:
                    raise RuntimeError("Failed to initialize PPO model. Check state dimensions and model parameters.")
            
        # Set model to evaluation mode for data collection
        self.policy.model.eval()
        
        # Collect experience
        episode_rewards = []
        episode_lengths = []
        total_episodes = 0
        current_ep_reward = 0
        current_ep_length = 0
        
        for step in range(num_steps):
            # Get state tensor
            state = JointState(self.robot.get_full_state(), obs)
            state_tensor = self.policy.get_state_tensor(state)
            
            # Get action, log_prob, and value
            with torch.no_grad():
                action_tensor, log_prob = self.policy.model.get_action(state_tensor)
                _, _, value = self.policy.model(state_tensor)
            
            # Convert action to numpy
            action_np = action_tensor.cpu().numpy().squeeze()
            
            # Clip action to max_action
            action_np = np.clip(action_np, -self.policy.max_action, self.policy.max_action)
            
            # Create ActionXY
            action = ActionXY(action_np[0], action_np[1])
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.buffer['states'].append(state_tensor)
            self.buffer['actions'].append(action_tensor.cpu())
            self.buffer['rewards'].append(torch.tensor([reward], dtype=torch.float32))
            self.buffer['values'].append(value.cpu())
            self.buffer['log_probs'].append(log_prob.cpu())
            self.buffer['dones'].append(torch.tensor([float(done)], dtype=torch.float32))
            
            # Update episode stats
            current_ep_reward += reward
            current_ep_length += 1
            
            # If episode ended
            if done:
                # Reset environment
                obs = self.env.reset(phase='train')
                
                # Store episode stats
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                total_episodes += 1
                
                # Reset episode stats
                current_ep_reward = 0
                current_ep_length = 0
        
        # Calculate average episode stats
        if total_episodes > 0:
            avg_reward = sum(episode_rewards) / total_episodes
            avg_length = sum(episode_lengths) / total_episodes
        else:
            avg_reward = current_ep_reward
            avg_length = current_ep_length
        
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'total_episodes': total_episodes
        }
    
    def compute_returns_and_advantages(self):
        """
        Compute returns and advantages for PPO
        """
        # Get device
        device = self.policy.device
        
        # Convert lists to tensors and move to device
        states = torch.cat(self.buffer['states']).to(device)
        rewards = torch.cat(self.buffer['rewards']).to(device)
        values = torch.cat(self.buffer['values']).to(device)
        dones = torch.cat(self.buffer['dones']).to(device)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_value = 0  # For terminal state
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
                
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]
            
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store in buffer
        self.buffer['advantages'] = advantages
        self.buffer['returns'] = returns
    
    def update_policy(self):
        """
        Update policy using PPO
        """
        # Get device
        device = self.policy.device
        
        # Convert buffer to tensors
        states = torch.cat(self.buffer['states']).to(device)
        actions = torch.cat(self.buffer['actions']).to(device)
        old_log_probs = torch.cat(self.buffer['log_probs']).to(device)
        advantages = self.buffer['advantages']
        returns = self.buffer['returns']
        
        # Create optimizer if not exists
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.policy.model.parameters(), lr=self.learning_rate)
        
        # Update for num_epochs
        for epoch in range(self.num_epochs):
            # Get batch indices
            indices = np.random.permutation(len(states))
            
            # Update in batches
            for start_idx in range(0, len(states), self.batch_size):
                # Get batch
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get new log probs and values
                new_log_probs, values, entropy = self.policy.model.evaluate_actions(batch_states, batch_actions)
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate objectives
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                
                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Return average losses
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train(self):
        """
        Train the PPO policy
        """
        logger.info(f"Starting training for {self.num_episodes} episodes on device: {self.policy.device}")
        
        # Training loop
        best_reward = -np.inf
        step_counter = 0
        
        for episode in range(1, self.num_episodes + 1):
            start_time = time.time()
            
            # Collect experience
            experience_stats = self.collect_experience(self.buffer_size)
            step_counter += self.buffer_size
            
            # Compute returns and advantages
            self.compute_returns_and_advantages()
            
            # Update policy
            update_stats = self.update_policy()
            
            # Log stats
            for key, value in update_stats.items():
                self.writer.add_scalar(f'train/{key}', value, step_counter)
            
            self.writer.add_scalar('train/avg_reward', experience_stats['avg_reward'], step_counter)
            self.writer.add_scalar('train/avg_episode_length', experience_stats['avg_length'], step_counter)
            
            # Save model if improved
            if experience_stats['avg_reward'] > best_reward:
                best_reward = experience_stats['avg_reward']
                self.save_model('best_model.pth')
            
            # Save checkpoint
            if episode % self.args.save_interval == 0:
                self.save_model(f'checkpoint_{episode}.pth')
            
            # Print progress
            elapsed_time = time.time() - start_time
            logger.info(f"Episode {episode}/{self.num_episodes} | " +
                       f"Steps: {step_counter} | " +
                       f"Avg Reward: {experience_stats['avg_reward']:.2f} | " +
                       f"Avg Length: {experience_stats['avg_length']:.2f} | " +
                       f"Episodes: {experience_stats['total_episodes']} | " +
                       f"Time: {elapsed_time:.2f}s")
        
        # Save final model
        self.save_model('final_model.pth')
        logger.info("Training completed!")
    
    def save_model(self, filename):
        """
        Save model to file
        
        Args:
            filename: Name of the file to save the model
        """
        path = os.path.join(self.save_dir, filename)
        self.policy.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def evaluate(self, num_episodes=100):
        """
        Evaluate the trained policy
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating policy for {num_episodes} episodes")
        
        # Load best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.policy.load_model(best_model_path)
        
        # Reset environment
        obs = self.env.reset(phase='test')
        
        # Evaluation metrics
        rewards = []
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        
        # Evaluate
        for episode in range(num_episodes):
            # Reset environment
            obs = self.env.reset(phase='test')
            done = False
            ep_reward = 0
            
            while not done:
                # Get state
                state = JointState(self.robot.get_full_state(), obs)
                
                # Get action
                action = self.policy.predict(state)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Update reward
                ep_reward += reward
            
            # Record metrics
            rewards.append(ep_reward)
            
            # Record outcome
            if info['event'] == 'reach_goal':
                success += 1
                success_times.append(self.env.global_time)
            elif info['event'] == 'collision':
                collision += 1
                collision_times.append(self.env.global_time)
            else:  # timeout
                timeout += 1
                timeout_times.append(self.env.time_limit)
            
            # Print progress
            if episode % 10 == 0:
                logger.info(f"Evaluated {episode}/{num_episodes} episodes | " +
                           f"Success: {success} | Collision: {collision} | Timeout: {timeout}")
        
        # Calculate average metrics
        success_rate = success / num_episodes
        collision_rate = collision / num_episodes
        timeout_rate = timeout / num_episodes
        avg_reward = sum(rewards) / num_episodes
        
        avg_success_time = sum(success_times) / len(success_times) if success_times else 0
        avg_collision_time = sum(collision_times) / len(collision_times) if collision_times else 0
        avg_timeout_time = sum(timeout_times) / len(timeout_times) if timeout_times else 0
        
        # Return metrics
        metrics = {
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_reward': avg_reward,
            'avg_success_time': avg_success_time,
            'avg_collision_time': avg_collision_time,
            'avg_timeout_time': avg_timeout_time
        }
        
        # Log metrics
        for key, value in metrics.items():
            self.writer.add_scalar(f'eval/{key}', value, 0)
        
        # Print metrics
        logger.info(f"Evaluation results:")
        logger.info(f"Success rate: {success_rate:.2f}")
        logger.info(f"Collision rate: {collision_rate:.2f}")
        logger.info(f"Timeout rate: {timeout_rate:.2f}")
        logger.info(f"Average reward: {avg_reward:.2f}")
        logger.info(f"Average success time: {avg_success_time:.2f}")
        logger.info(f"Average collision time: {avg_collision_time:.2f}")
        logger.info(f"Average timeout time: {avg_timeout_time:.2f}")
        
        return metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train PPO policy for robot navigation")
    
    # Training parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save models and logs')
    parser.add_argument('--exp_name', type=str, default='ppo_experiment', help='Experiment name')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=2048, help='Buffer size for experience collection')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_param', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--save_interval', type=int, default=100, help='Save interval')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model after training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'], help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Create trainer
    trainer = PPOTrainer(args)
    
    # Set device for policy
    trainer.policy.set_device(torch.device(args.device))
    
    # Initialize the model by getting state dimension from a dummy environment reset
    # This ensures the model is built before training starts
    obs = trainer.env.reset(phase='train')
    state = JointState(trainer.robot.get_full_state(), obs)
    _ = trainer.policy.get_state_tensor(state)  # This will trigger model building
    
    # Explicitly ensure the model is initialized
    if trainer.policy.model is None:
        trainer.policy._ensure_model_initialized(state)
        if trainer.policy.model is None:
            raise RuntimeError("Failed to initialize PPO model. Check state dimensions and model parameters.")
            
    # Reset environment again to start fresh
    trainer.env.reset(phase='train')
    
    # Train
    trainer.train()
    
    # Evaluate
    if args.eval:
        trainer.evaluate() 