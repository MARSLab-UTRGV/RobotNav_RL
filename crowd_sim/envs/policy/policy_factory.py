from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.random import RandomPolicy
from crowd_sim.envs.policy.ppo_policy import PPOPolicy


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['random'] = RandomPolicy
policy_factory['ppo'] = PPOPolicy
policy_factory['none'] = none_policy
# Transformer policy will be registered by the transformer_policy module
