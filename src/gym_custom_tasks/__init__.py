from gym.envs.registration import register

register(
    id='SimplePathing-v0',
    entry_point='src.gym_custom_tasks.envs:SimplePathing',
    kwargs={'width': 30, 'height': 30, 'visual': False},
)
register(
    id='VisualSimplePathing-v0',
    entry_point='src.gym_custom_tasks.envs:SimplePathing',
    kwargs={'width': 30, 'height': 30, 'visual': True},
)
register(
    id='ObstaclePathing-v0',
    entry_point='src.gym_custom_tasks.envs:ObstaclePathing',
    kwargs={'width': 30, 'height': 30,
            'obstacles': [[0, 18, 18, 21],
                          [21, 24, 10, 30]],
            'visual': False},
)
register(
    id='VisualObstaclePathing-v0',
    entry_point='src.gym_custom_tasks.envs:ObstaclePathing',
    kwargs={'width':30, 'height':30,
            'obstacles':[[0, 18, 18, 21],
                        [21, 24, 10, 30]],
            'visual': True},
)
register(
    id='Race-v0',
    entry_point='src.gym_custom_tasks.envs:Race',
    kwargs={'width':10, 'height':10,
            'driver_chance':0.01},
)
register(
    id='Evasion-v0',
    entry_point='src.gym_custom_tasks.envs:Evasion',
    kwargs={'width':10, 'height':10,
            'obstacle_chance':0.01},
)