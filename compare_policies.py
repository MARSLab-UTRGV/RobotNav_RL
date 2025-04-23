import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import configparser

from sim import RobotSimulator
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.ppo_policy import PPOPolicy
from crowd_sim.envs.policy.random import RandomPolicy
from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.utils.state import JointState


def create_comparison_config():
    """Create a configuration for comparison simulations."""
    config = configparser.ConfigParser()

    # Environment settings
    config.add_section('env')
    config.set('env', 'time_limit', '25')
    config.set('env', 'time_step', '0.25')
    config.set('env', 'randomize_attributes', 'False')
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
    config.set('robot', 'policy', 'none')  # Will be set in the simulator
    config.set('robot', 'radius', '0.3')
    config.set('robot', 'v_pref', '1')
    config.set('robot', 'visible', 'True')
    config.set('robot', 'sensor', 'coordinates')

    # PPO policy settings
    config.add_section('ppo')
    config.set('ppo', 'hidden_dim', '128')

    # Random policy settings
    config.add_section('random')
    config.set('random', 'max_speed', '1.0')
    config.set('random', 'random_seed', '42')

    return config


def run_evaluation(policy_name, model_path=None, num_episodes=100, visualize=True, seed=42):
    """
    Run evaluation for a specific policy
    
    Args:
        policy_name: Name of the policy ('orca', 'ppo', 'random', or 'linear')
        model_path: Path to the model for PPO policy
        num_episodes: Number of episodes to evaluate
        visualize: Whether to visualize the simulation
        seed: Random seed for reproducibility
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n=== Evaluating {policy_name.upper()} policy ===")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create config
    config = create_comparison_config()
    
    # Set up simulator
    simulator = RobotSimulator(config=config)
    
    # Add model path to config if PPO
    if policy_name == 'ppo' and model_path:
        config.set('ppo', 'model_path', model_path)
    
    # Set up environment with the selected policy
    simulator.setup(policy_type=policy_name)
    
    # Set device for PPO policy
    if policy_name == 'ppo':
        simulator.robot.policy.set_device(torch.device('cuda'))
        
        # Initialize the model if using PPO
        # This will trigger model building if not already built
        state = JointState(simulator.robot.get_full_state(), simulator.ob)
        _ = simulator.robot.policy.get_state_tensor(state)
        
        # Force model initialization
        simulator.robot.policy._ensure_model_initialized(state)
        
        # Check if model was initialized
        if simulator.robot.policy.model is None:
            raise RuntimeError("Failed to initialize PPO model. Check model path and parameters.")
            
        print(f"PPO model initialized successfully on {'cuda'}")
    
    # Set up visualization if requested
    if visualize:
        simulator.setup_visualization()
    
    # Evaluation metrics
    metrics = {
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'rewards': [],
        'success_times': [],
        'collision_times': [],
        'timeout_times': [],
        'path_lengths': [],
        'min_human_distances': []
    }
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        # Reset simulator for each episode
        simulator.setup(policy_type=policy_name)
        
        # Track minimum distance to humans in this episode
        min_distance_to_human = float('inf')
        path_length = 0
        step_count = 0
        
        # Run episode
        done = False
        ep_reward = 0
        
        # Start timer
        start_time = time.time()
        
        while not done:
            # Step simulator
            _, reward, terminated, truncated, info = simulator.step()
            done = terminated or truncated
            
            # Update reward
            ep_reward += reward
            
            # Track path length (distance traveled)
            if step_count > 0:
                robot_state = simulator.robot.get_full_state()
                # Check if get_prev_state method exists, otherwise use a different approach
                if hasattr(simulator.robot, 'get_prev_state'):
                    prev_robot_state = simulator.robot.get_prev_state()
                    if prev_robot_state is not None:
                        dx = robot_state.px - prev_robot_state.px
                        dy = robot_state.py - prev_robot_state.py
                        path_length += np.sqrt(dx**2 + dy**2)
                else:
                    # Alternative approach: use current position and velocity to estimate path length
                    # This is an approximation for one step
                    path_length += np.sqrt(robot_state.vx**2 + robot_state.vy**2) * simulator.env.time_step
            
            # Track minimum distance to humans
            for human in simulator.env.humans:
                robot_state = simulator.robot.get_full_state()
                dx = human.px - robot_state.px
                dy = human.py - robot_state.py
                dist = np.sqrt(dx**2 + dy**2) - human.radius - robot_state.radius
                min_distance_to_human = min(min_distance_to_human, dist)
            
            # Update visualization
            if visualize:
                simulator.update_visualization()
                plt.pause(0.01)
            
            step_count += 1
        
        # End timer
        episode_time = time.time() - start_time
        
        # Record metrics
        metrics['rewards'].append(ep_reward)
        metrics['path_lengths'].append(path_length)
        
        if min_distance_to_human != float('inf'):
            metrics['min_human_distances'].append(min_distance_to_human)
        
        # Record outcome
        if info['event'] == 'reach_goal':
            metrics['success'] += 1
            metrics['success_times'].append(simulator.env.global_time)
        elif info['event'] == 'collision':
            metrics['collision'] += 1
            metrics['collision_times'].append(simulator.env.global_time)
        else:  # timeout
            metrics['timeout'] += 1
            metrics['timeout_times'].append(simulator.env.time_limit)
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == num_episodes - 1:
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Success: {metrics['success']} | "
                  f"Collision: {metrics['collision']} | "
                  f"Timeout: {metrics['timeout']} | "
                  f"Time: {episode_time:.2f}s")
    
    # Calculate final metrics
    metrics['success_rate'] = metrics['success'] / num_episodes
    metrics['collision_rate'] = metrics['collision'] / num_episodes
    metrics['timeout_rate'] = metrics['timeout'] / num_episodes
    metrics['avg_reward'] = np.mean(metrics['rewards'])
    metrics['avg_path_length'] = np.mean(metrics['path_lengths']) if metrics['path_lengths'] else 0
    metrics['avg_min_human_distance'] = np.mean(metrics['min_human_distances']) if metrics['min_human_distances'] else 0
    
    if metrics['success_times']:
        metrics['avg_success_time'] = np.mean(metrics['success_times'])
    if metrics['collision_times']:
        metrics['avg_collision_time'] = np.mean(metrics['collision_times'])
    if metrics['timeout_times']:
        metrics['avg_timeout_time'] = np.mean(metrics['timeout_times'])
    
    # Print final metrics
    print(f"\n=== Results for {policy_name.upper()} policy ===")
    print(f"Success rate: {metrics['success_rate']:.2f}")
    print(f"Collision rate: {metrics['collision_rate']:.2f}")
    print(f"Timeout rate: {metrics['timeout_rate']:.2f}")
    print(f"Average reward: {metrics['avg_reward']:.2f}")
    print(f"Average path length: {metrics['avg_path_length']:.2f}")
    print(f"Average minimum distance to humans: {metrics['avg_min_human_distance']:.2f}")
    if 'avg_success_time' in metrics:
        print(f"Average success time: {metrics['avg_success_time']:.2f}")
    if 'avg_collision_time' in metrics:
        print(f"Average collision time: {metrics['avg_collision_time']:.2f}")
    if 'avg_timeout_time' in metrics:
        print(f"Average timeout time: {metrics['avg_timeout_time']:.2f}")
    
    # Close simulator
    simulator.close()
    
    return metrics


def compare_policies(ppo_model_path, num_episodes=50, visualize=True, seed=42):
    """
    Compare different policies
    
    Args:
        ppo_model_path: Path to the trained PPO model
        num_episodes: Number of episodes to evaluate
        visualize: Whether to visualize the simulation
        seed: Random seed for reproducibility
    """
    results = {}
    
    # Evaluate ORCA policy
    results['orca'] = run_evaluation('orca', num_episodes=num_episodes, visualize=visualize, seed=seed)
    
    # Evaluate PPO policy
    results['ppo'] = run_evaluation('ppo', model_path=ppo_model_path, num_episodes=num_episodes, visualize=visualize, seed=seed)
    
    # Compare results
    print("\n=== Policy Comparison ===")
    policies = list(results.keys())
    
    # Compare success rates
    success_rates = [results[policy]['success_rate'] for policy in policies]
    print("\nSuccess Rates:")
    for i, policy in enumerate(policies):
        print(f"{policy.upper()}: {success_rates[i]:.2f}")
    
    # Compare collision rates
    collision_rates = [results[policy]['collision_rate'] for policy in policies]
    print("\nCollision Rates:")
    for i, policy in enumerate(policies):
        print(f"{policy.upper()}: {collision_rates[i]:.2f}")
    
    # Compare average rewards
    avg_rewards = [results[policy]['avg_reward'] for policy in policies]
    print("\nAverage Rewards:")
    for i, policy in enumerate(policies):
        print(f"{policy.upper()}: {avg_rewards[i]:.2f}")
    
    # Compare path lengths
    avg_paths = [results[policy]['avg_path_length'] for policy in policies]
    print("\nAverage Path Lengths:")
    for i, policy in enumerate(policies):
        print(f"{policy.upper()}: {avg_paths[i]:.2f}")
    
    # Compare minimum distances to humans
    avg_min_dists = [results[policy]['avg_min_human_distance'] for policy in policies]
    print("\nAverage Minimum Distance to Humans:")
    for i, policy in enumerate(policies):
        print(f"{policy.upper()}: {avg_min_dists[i]:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Success, collision, timeout rates
    plt.subplot(2, 2, 1)
    width = 0.25
    x = np.arange(len(policies))
    plt.bar(x - width, [results[policy]['success_rate'] for policy in policies], width, label='Success')
    plt.bar(x, [results[policy]['collision_rate'] for policy in policies], width, label='Collision')
    plt.bar(x + width, [results[policy]['timeout_rate'] for policy in policies], width, label='Timeout')
    plt.xticks(x, [p.upper() for p in policies])
    plt.ylabel('Rate')
    plt.title('Success, Collision, and Timeout Rates')
    plt.legend()
    
    # Average rewards
    plt.subplot(2, 2, 2)
    plt.bar(policies, avg_rewards)
    plt.xticks([p.upper() for p in policies])
    plt.ylabel('Average Reward')
    plt.title('Average Rewards')
    
    # Path lengths
    plt.subplot(2, 2, 3)
    plt.bar(policies, avg_paths)
    plt.xticks([p.upper() for p in policies])
    plt.ylabel('Average Path Length')
    plt.title('Average Path Lengths')
    
    # Minimum distances to humans
    plt.subplot(2, 2, 4)
    plt.bar(policies, avg_min_dists)
    plt.xticks([p.upper() for p in policies])
    plt.ylabel('Average Minimum Distance')
    plt.title('Average Minimum Distance to Humans')
    
    plt.tight_layout()
    plt.savefig('policy_comparison.png')
    plt.show()
    
    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare navigation policies")
    
    parser.add_argument('--ppo_model', type=str, default='output/ppo_experiment/best_model.pth',
                        help='Path to trained PPO model')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to evaluate')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--single', type=str, default=None, choices=['orca', 'ppo', 'random', 'linear'],
                        help='Evaluate only a single policy')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Check if PPO model exists
    if args.single == 'ppo' or args.single is None:
        if not os.path.exists(args.ppo_model):
            print(f"PPO model not found at {args.ppo_model}")
            print("Please train a PPO model first or specify a different path with --ppo_model")
            exit(1)
    
    # Evaluate single policy or compare
    if args.single:
        run_evaluation(args.single, 
                      model_path=args.ppo_model if args.single == 'ppo' else None,
                      num_episodes=args.episodes, 
                      visualize=not args.no_vis, 
                      seed=args.seed)
    else:
        compare_policies(args.ppo_model, 
                        num_episodes=args.episodes, 
                        visualize=not args.no_vis, 
                        seed=args.seed)