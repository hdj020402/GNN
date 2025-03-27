def calc_mean_std(dataset):
    sum = 0
    for i in dataset:
        sum += i
    mean = sum / len(dataset)
    
    _sum = 0
    for i in dataset:
        _sum += (i - mean) ** 2
    std = (_sum / len(dataset)) ** 0.5
    return mean, std
