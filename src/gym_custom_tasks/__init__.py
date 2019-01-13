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

# VISUAL OBSTACLE PATHING

register(
    id='VisualObstaclePathing-v0',
    entry_point='src.gym_custom_tasks.envs:ObstaclePathing',
    kwargs={'width':30, 'height':30,
            'obstacles':[[0, 18, 18, 21],
                        [21, 24, 10, 30]],
            'visual': True},
)

register(
    id='VisualObstaclePathing-v1',
    entry_point='src.gym_custom_tasks.envs:ObstaclePathing',
    kwargs={'width':30, 'height':30,
            'obstacles':[[5, 24, 15, 19],
                        [16, 19, 5, 23]],
            'visual': True},
)

register(
    id='VisualObstaclePathing-v2',
    entry_point='src.gym_custom_tasks.envs:ObstaclePathing',
    kwargs={'width':30, 'height':30,
            'obstacles':[[7, 15, 7, 10],
                         [7, 15, 14, 17],
                         [7, 25, 21, 24],
                         [19, 24, 7, 17]],
            'visual': True},
)

register(
    id='VisualObstaclePathing-v3',
    entry_point='src.gym_custom_tasks.envs:ObstaclePathing',
    kwargs={'width':30, 'height':30,
            'obstacles':[[7, 10, 7, 15],
                         [14, 17, 7, 15],
                         [21, 24, 7, 25],
                         [7, 17, 19, 24]],
            'visual': True},
)

# RACE / EVASION

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
