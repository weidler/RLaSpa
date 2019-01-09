from gym.envs.registration import register

register(
    id='SimplePathing-v0',
    entry_point='src.gym_pathing.envs:SimplePathing',
    kwargs={'width': 30, 'height': 30, 'visual': False},
)
register(
    id='ObstaclePathing-v0',
    entry_point='src.gym_pathing.envs:ObstaclePathing',
    kwargs={'width': 30, 'height': 30,
            'obstacles': [[0, 18, 18, 21],
                          [21, 24, 10, 30]],
            'visual': False},
)
register(
    id='VisualObstaclePathing-v0',
    entry_point='src.gym_pathing.envs:ObstaclePathing',
    kwargs={'width':30, 'height':30,
            'obstacles':[[0, 18, 18, 21],
                        [21, 24, 10, 30]],
            'visual': True},
)