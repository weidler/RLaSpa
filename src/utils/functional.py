def int_to_one_hot(a: int, i_max: int):
    return [0 if i != a else 1 for i in range(i_max)]