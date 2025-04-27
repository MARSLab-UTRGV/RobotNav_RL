import configparser
import numpy as np
import sys

from crowd_sim.envs import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.random import RandomPolicy
from crowd_sim.envs.policy.linear import Linear

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import logging

# Import the self-attention model and training function
import torch
from self_attention import TrajectorySelfAttention, train_model

class RobotSimulatorWithPred:
    """
    A class to set up and run robot navigation simulations in a crowd environment.
    This simulator provides real-time visualization, predicts human future positions using self-attention,
    and displays predictions as circles centered on the predicted positions for each future step.
    """
    
    def __init__(self, config=None):
        """
        Initialize the simulator with optional custom configuration.
        
        Args:
            config: Optional configparser.ConfigParser object with simulation settings.
                   If None, default configuration will be created.
        """
        # Initialize simulation configuration
        self.config = config if config is not None else self._create_default_config()
        
        # Initialize environment, robot, and visualization components
        self.env = None
        self.robot = None
        self.robot_policy = None
        self.fig = None
        self.ax1 = None  # Simulation view
        self.ax2 = None  # Information panel
        self.visualization_components = {}  # Will store visual elements
        
        # Simulation state variables
        self.step_count = 0
        self.done = False
        self.ob = None
        self.info = None
        self.reward = 0

        # Trajectory prediction setup
        self.seq_len = 5  # Past 5 steps
        self.future_steps = 5  # Predict 5 future steps
        self.human_num = int(self.config.get('sim', 'human_num'))  # Number of humans
        self.trajectory_buffer = np.zeros((self.seq_len, self.human_num, 2))  # Buffer for past positions
        self.predicted_positions = np.zeros((self.human_num, self.future_steps, 2))  # Predicted positions

        # Load the self-attention model
        self.model = TrajectorySelfAttention(max_humans=self.human_num)
        try:
            self.model.load_state_dict(torch.load("trajectory_model.pth"))
            self.model.eval()
            print("Loaded self-attention model for trajectory prediction.")
        except FileNotFoundError:
            print("Warning: trajectory_model.pth not found. Collecting data and training the model...")
            # Collect data and train the model
            data, labels = self.collect_simulation_data(num_episodes=100)
            self.train_model(data, labels)
            self.model.load_state_dict(torch.load("trajectory_model.pth"))
            self.model.eval()
            print("Model training completed and loaded.")
        
    def _create_default_config(self):
        """Create a default configuration for the simulation."""
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
        config.set('robot', 'policy', 'orca')
        config.set('robot', 'radius', '0.3')
        config.set('robot', 'v_pref', '1')
        config.set('robot', 'visible', 'True')
        config.set('robot', 'sensor', 'coordinates')

        # Random policy settings
        config.add_section('random')
        config.set('random', 'max_speed', '1.0')
        config.set('random', 'random_seed', '42')

        # Transformer settings
        config.add_section('transformer')
        config.set('transformer', 'model_path', 'output/transformer_experiment/best_model.pth')
        config.set('transformer', 'use_self_attn', 'True')
        config.set('transformer', 'num_attention_heads', '8')

        return config

    def collect_simulation_data(self, num_episodes=100, policy_type='orca'):
        """Collect trajectory data from the CrowdSim simulation."""
        data = []
        labels = []
        total_steps = 0

        # Buffer to store positions for the current episode
        max_steps_per_episode = int(float(self.config.get('env', 'time_limit')) / float(self.config.get('env', 'time_step')))  # 25 / 0.25 = 100 steps
        trajectory_buffer = np.zeros((max_steps_per_episode, self.human_num, 2))

        for episode in range(num_episodes):
            print(f"Collecting data from episode {episode + 1}/{num_episodes}")
            self.setup(policy_type=policy_type)  # Reset the simulation
            step_idx = 0
            while not self.done and step_idx < max_steps_per_episode:
                # Collect current human positions
                for i, human in enumerate(self.env.humans):
                    trajectory_buffer[step_idx, i] = [human.px, human.py]

                # Step the simulation
                self.step()
                step_idx += 1

                # Collect data sample if we have enough steps
                if step_idx >= self.seq_len + self.future_steps:
                    for t in range(step_idx - (self.seq_len + self.future_steps) + 1):
                        past = trajectory_buffer[t:t + self.seq_len]  # [seq_len, num_humans, 2]
                        future = trajectory_buffer[t + self.seq_len:t + self.seq_len + self.future_steps]  # [future_steps, num_humans, 2]
                        data.append(past)
                        labels.append(future)

            total_steps += step_idx

        print(f"Collected {len(data)} samples from {total_steps} total steps across {num_episodes} episodes.")
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def train_model(self, data, labels, num_epochs=50, batch_size=32):
        """Train the self-attention model with collected data."""
        train_model(self.model, data, labels, num_epochs=num_epochs, batch_size=batch_size)

    def setup(self, policy_type='orca'):
        """
        Set up the simulation environment and robot.
        
        Args:
            policy_type: Type of robot policy to use ('orca', 'random', 'linear', 'ppo', 'transformer')
        """
        # Create the crowd simulation environment
        self.env = CrowdSim()
        self.env.configure(self.config)
        
        # Create robot with specified policy
        if policy_type.lower() == 'orca':
            self.robot_policy = ORCA()
        elif policy_type.lower() == 'random':
            self.robot_policy = RandomPolicy()
        elif policy_type.lower() == 'linear':
            self.robot_policy = Linear()
        elif policy_type.lower() == 'ppo':
            from crowd_sim.envs.policy.ppo_policy import PPOPolicy
            self.robot_policy = PPOPolicy()
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}. Choose 'orca', 'random', 'linear', 'ppo', or 'transformer'.")
        
        # Update config to match selected policy
        self.config.set('robot', 'policy', policy_type.lower())
            
        # Create and configure robot
        self.robot = Robot(self.config, 'robot')
        self.robot.set_policy(self.robot_policy)

        # Set the robot in the environment
        self.env.set_robot(self.robot)

        # Reset the environment
        self.ob = self.env.reset(phase='test')
        
        # Reset simulation state variables
        self.step_count = 0
        self.done = False
        self.info = None
        self.reward = 0

        # Initialize trajectory buffer with current human positions
        for i, human in enumerate(self.env.humans):
            self.trajectory_buffer[:, i, :] = [human.px, human.py]  # Fill with initial position
        
        return self
    
    def setup_visualization(self):
        """Set up the visualization for real-time rendering of the simulation."""
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [3, 1]})
        
        # Configure the simulation view
        self.ax1.set_xlim(-10, 10)
        self.ax1.set_ylim(-10, 10)
        self.ax1.set_xlabel('x(m)', fontsize=14)
        self.ax1.set_ylabel('y(m)', fontsize=14)
        self.ax1.set_title('Simulation', fontsize=16)
        
        # Configure the info panel
        self.ax2.axis('off')
        self.ax2.set_title('Robot Information', fontsize=16)
        
        # Initialize visualization components dictionary
        vc = self.visualization_components
        
        # Add goal marker
        vc['goal'] = plt.Line2D([0], [4], color='red', marker='*', linestyle='None', 
                          markersize=15, label='Goal')
        self.ax1.add_artist(vc['goal'])
        
        # Add initial robot circle
        robot_state = self.robot.get_full_state()
        vc['robot_circle'] = plt.Circle(
            (robot_state.px, robot_state.py), 
            self.robot.radius, 
            fill=True, 
            color='yellow',
            label='Robot'
        )
        self.ax1.add_artist(vc['robot_circle'])
        
        # Add velocity arrow for robot
        arrow_length_factor = 0.5
        vc['robot_vel_arrow'] = self.ax1.arrow(
            robot_state.px, 
            robot_state.py, 
            robot_state.vx * arrow_length_factor, 
            robot_state.vy * arrow_length_factor,
            width=0.05, 
            head_width=0.2, 
            head_length=0.2, 
            fc='red', 
            ec='red'
        )
        
        # Initial human circles and prediction rings
        vc['human_circles'] = []
        vc['human_vel_arrows'] = []
        vc['prediction_rings'] = []  # List of lists: one list per human, containing rings for future steps
        for i, human in enumerate(self.env.humans):
            # Human circle
            human_circle = plt.Circle(
                (human.px, human.py), 
                human.radius, 
                fill=False, 
                color='blue'
            )
            self.ax1.add_artist(human_circle)
            vc['human_circles'].append(human_circle)
            
            # Human velocity arrow
            human_vel_arrow = self.ax1.arrow(
                human.px, 
                human.py, 
                human.vx * arrow_length_factor, 
                human.vy * arrow_length_factor,
                width=0.03, 
                head_width=0.15, 
                head_length=0.15, 
                fc='green', 
                ec='green'
            )
            vc['human_vel_arrows'].append(human_vel_arrow)
            
            # Prediction rings (initially centered on human's current position; will be updated later)
            human_rings = []
            for step in range(self.future_steps):
                ring = plt.Circle(
                    (human.px, human.py),
                    radius=0.3,  # Smaller radii: 0.1, 0.2, 0.3, 0.4, 0.5
                    fill=False,
                    linestyle='--',
                    color='orange',
                    alpha=0.5
                )
                self.ax1.add_artist(ring)
                human_rings.append(ring)
            vc['prediction_rings'].append(human_rings)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='lightblue', label='Human in radius'),
            Patch(facecolor='blue', edgecolor='blue', label='Other humans'),
            Patch(facecolor='purple', edgecolor='purple', linestyle='--', label='Predicted positions'),
            vc['robot_circle'],
            vc['goal']
        ]
        self.ax1.legend(handles=legend_elements, loc='upper right')
        
        # Text for robot info
        vc['info_text'] = self.ax2.text(
            0.1, 0.9, 
            "", 
            fontsize=12, 
            verticalalignment='top', 
            transform=self.ax2.transAxes
        )
        
        # Turn on interactive mode
        plt.ion()
        plt.show(block=False)
        
        # Initial info text update
        self._update_info_text()
        
        return self
        
    def _update_info_text(self):
        """Update the information text in the visualization."""
        if 'info_text' not in self.visualization_components:
            return
            
        robot_state = self.robot.get_full_state()
        robot_info = f"Step: {self.step_count}\n\n"
        robot_info += f"Position: ({robot_state.px:.2f}, {robot_state.py:.2f})\n"
        robot_info += f"Velocity: ({robot_state.vx:.2f}, {robot_state.vy:.2f})\n"
        robot_info += f"Goal: ({robot_state.gx:.2f}, {robot_state.gy:.2f})\n"
        robot_info += f"Distance to goal: {np.linalg.norm([robot_state.px-robot_state.gx, robot_state.py-robot_state.gy]):.2f}\n"
        robot_info += f"Policy: {self.robot_policy.name}\n"
        
        if self.info:
            robot_info += f"\nEvent: {self.info.get('event', 'unknown')}\n"
            robot_info += f"Reward: {self.reward:.2f}"
            
        self.visualization_components['info_text'].set_text(robot_info)
    
    def predict_human_positions(self):
        """Use the self-attention model to predict future human positions."""
        if self.model is None:
            return

        # Prepare input for the model: [1, seq_len, num_humans, 2]
        input_data = torch.tensor(self.trajectory_buffer, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(input_data)  # [1, num_humans, future_steps, 2]
        self.predicted_positions = pred.squeeze(0).numpy()  # [num_humans, future_steps, 2]

    def step(self):
        """
        Execute a single step of the simulation.
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if self.done:
            print("Simulation already completed. Call setup() to reset.")
            return None, None, True, True, self.info
            
        # Robot action
        action = self.robot.act(self.ob)
        self.ob, self.reward, terminated, truncated, self.info = self.env.step(action)
        
        # Update trajectory buffer
        self.trajectory_buffer[:-1] = self.trajectory_buffer[1:]  # Shift past positions
        for i, human in enumerate(self.env.humans):
            self.trajectory_buffer[-1, i] = [human.px, human.py]

        # Predict future positions
        self.predict_human_positions()
        
        # Update step count
        self.step_count += 1
        
        # Check if simulation is done
        self.done = terminated or truncated
        
        # Return step results
        return self.ob, self.reward, terminated, truncated, self.info
    
    def update_visualization(self):
        """Update the visualization based on current simulation state, including predicted positions."""
        if not hasattr(self, 'fig') or self.fig is None:
            print("Visualization not set up. Call setup_visualization() first.")
            return self
            
        vc = self.visualization_components
        if not vc:
            return self
            
        # Get current robot state
        robot_state = self.robot.get_full_state()
        
        # Update robot position
        vc['robot_circle'].center = (robot_state.px, robot_state.py)
        
        # Add/update robot attention radius (dotted circle)
        attention_radius = 2.0
        if 'attention_radius' not in vc:
            vc['attention_radius'] = plt.Circle(
                (robot_state.px, robot_state.py),
                attention_radius,
                fill=False,
                linestyle='--',
                color='orange',
                alpha=0.7
            )
            self.ax1.add_artist(vc['attention_radius'])
        else:
            vc['attention_radius'].center = (robot_state.px, robot_state.py)
        
        # Remove old velocity arrow and create a new one
        arrow_length_factor = 0.5
        vc['robot_vel_arrow'].remove()
        vc['robot_vel_arrow'] = self.ax1.arrow(
            robot_state.px, 
            robot_state.py, 
            robot_state.vx * arrow_length_factor, 
            robot_state.vy * arrow_length_factor,
            width=0.05, 
            head_width=0.2, 
            head_length=0.2, 
            fc='red', 
            ec='red'
        )
        
        # Get attention weights from policy if available
        attention_weights = None
        if hasattr(self.robot.policy, 'model') and hasattr(self.robot.policy.model, 'human_robot_attention'):
            attention_weights = self.robot.policy.model.human_robot_attention.attn_weights
            if attention_weights is not None:
                attention_weights = attention_weights.mean(dim=1).squeeze().cpu().numpy()
        
        # Update human positions, velocities, and prediction rings
        for i, human in enumerate(self.env.humans):
            # Update human circle
            vc['human_circles'][i].center = (human.px, human.py)
            
            # Calculate distance to robot
            dist_to_robot = np.sqrt((human.px - robot_state.px)**2 + (human.py - robot_state.py)**2)
            
            # Determine if human is relevant based on attention weights
            is_relevant = False
            if attention_weights is not None and i < len(attention_weights):
                is_relevant = attention_weights[i] > 0.2
            
            # Change color based on relevance and distance
            if is_relevant:
                if dist_to_robot <= attention_radius:
                    vc['human_circles'][i].set_color('red')
                    vc['human_circles'][i].set_alpha(1.0)
                    vc['human_circles'][i].set_linewidth(2.0)
                else:
                    vc['human_circles'][i].set_color('orange')
                    vc['human_circles'][i].set_alpha(0.8)
                    vc['human_circles'][i].set_linewidth(1.5)
            else:
                if dist_to_robot <= attention_radius:
                    vc['human_circles'][i].set_color('lightblue')
                    vc['human_circles'][i].set_alpha(0.8)
                else:
                    vc['human_circles'][i].set_color('blue')
                    vc['human_circles'][i].set_alpha(0.6)
            
            # Remove old velocity arrow and create a new one
            vc['human_vel_arrows'][i].remove()
            vc['human_vel_arrows'][i] = self.ax1.arrow(
                human.px, 
                human.py, 
                human.vx * arrow_length_factor, 
                human.vy * arrow_length_factor,
                width=0.03, 
                head_width=0.15, 
                head_length=0.15, 
                fc='green', 
                ec='green'
            )
            
            # Update prediction rings (centered on the predicted position for each future step)
            for step in range(self.future_steps):
                predicted_pos = self.predicted_positions[i, step]  # [x, y] for this human and step
                vc['prediction_rings'][i][step].center = (predicted_pos[0], predicted_pos[1])
        
        # Update info text
        self._update_info_text()
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        return self
    
    def run(self, max_steps=None, step_delay=0.05, print_step_info=True):
        """
        Run the full simulation until completion or max_steps is reached.
        
        Args:
            max_steps: Maximum number of steps to run (None for unlimited)
            step_delay: Delay between steps for visualization (seconds)
            print_step_info: Whether to print step information to console
            
        Returns:
            dict: Final simulation info
        """
        if self.env is None:
            print("Environment not set up. Call setup() first.")
            return None
            
        while not self.done:
            if max_steps is not None and self.step_count >= max_steps:
                break
                
            # Execute step
            self.ob, self.reward, terminated, truncated, self.info = self.step()
            
            # Print step information if requested
            if print_step_info:
                robot_state = self.robot.get_full_state()
                print(f'\nStep {self.step_count} - Robot State:')
                print(f' Position: ({robot_state.px:.2f}, {robot_state.py:.2f})')
                print(f' Velocity: ({robot_state.vx:.2f}, {robot_state.vy:.2f})')
                print(f' Event: {self.info["event"]}')
            
            # Update visualization
            self.update_visualization()
            
            # Add delay for visualization
            plt.pause(step_delay)
            
        # Print completion message
        print(f"\nSimulation completed after {self.step_count} steps")
        print(f"Final event: {self.info['event']}")
        
        # Keep plot open until closed
        if hasattr(self, 'fig') and self.fig is not None:
            plt.ioff()
            plt.show()
            
        return self.info
            
    def close(self):
        """Close the simulation and release resources."""
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            
        if self.env is not None:
            self.env.close()
            
        return self

if __name__ == '__main__':
    # Get Model Name
    if len(sys.argv) < 2:
        print('No Model Provided', file=sys.stderr)
        sys.exit(1)
    else:
        policy = sys.argv[1]
        # Create simulator with default configuration
        simulator = RobotSimulatorWithPred()
        
        # Set up environment with policy
        simulator.setup(policy_type=policy)
        
        # Set up visualization
        simulator.setup_visualization()
        
        # Run the simulation
        simulator.run(step_delay=0.10)
        
        # Close the simulator
        simulator.close()