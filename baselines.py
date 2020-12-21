from spinup import ppo_pytorch as ppo
from spinup import ddpg_pytorch as ddpg
from spinup import sac_pytorch as sac
import tensorflow as tf
import gym
import torch
"""
env_fn = lambda : gym.make('Walker2d-v2')
ac_kwargs = dict(hidden_sizes=[64,64])
logger_kwargs = dict(output_dir='baseline_data/walker/ppo', exp_name='walker_ppo')
ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

#env_fn = lambda : gym.make('Walker2d-v2')
#ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='baseline_data/walker/ddpg', exp_name='walker_ddpg')
ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

logger_kwargs = dict(output_dir='baseline_data/walker/sac', exp_name='walker_sac')
sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

env_fn = lambda : gym.make('Hopper-v2')
logger_kwargs = dict(output_dir='baseline_data/hopper/ppo', exp_name='hopper_ppo')
ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

# env_fn = lambda : gym.make('Walker2d-v2')
# ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='baseline_data/hopper/ddpg', exp_name='hopper_ddpg')
ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

logger_kwargs = dict(output_dir='baseline_data/hopper/sac', exp_name='hopper_sac')
sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)
"""
# env_fn = lambda : gym.make('HalfCheetah-v2')
ac_kwargs = dict(hidden_sizes=[64,64])
# logger_kwargs = dict(output_dir='baseline_data/HalfCheetah/ppo', exp_name='HalfCheetah_ppo')
# ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

# # env_fn = lambda : gym.make('Walker2d-v2')
# # ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)
# logger_kwargs = dict(output_dir='baseline_data/HalfCheetah/ddpg', exp_name='HalfCheetah_ddpg')
# ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

# logger_kwargs = dict(output_dir='baseline_data/HalfCheetah/sac', exp_name='HalfCheetah_sac')
# sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

# env_fn = lambda : gym.make('Ant-v2')
# logger_kwargs = dict(output_dir='baseline_data/Ant/ppo', exp_name='Ant_ppo')
# ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

# # env_fn = lambda : gym.make('Walker2d-v2')
# # ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)
# logger_kwargs = dict(output_dir='baseline_data/Ant/ddpg', exp_name='Ant_ddpg')
# ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

# logger_kwargs = dict(output_dir='baseline_data/Ant/sac', exp_name='Ant_sac')
# sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)


env_fn = lambda : gym.make('Humanoid-v2')
# logger_kwargs = dict(output_dir='baseline_data/Humanoid/ppo', exp_name='Humanoid_ppo')
# ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

# env_fn = lambda : gym.make('Walker2d-v2')
# ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='baseline_data/Humanoid/ddpg', exp_name='Humanoid_ddpg')
ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

logger_kwargs = dict(output_dir='baseline_data/Humanoid/sac', exp_name='Humanoid_sac')
sac(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)

