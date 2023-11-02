def datasetpath(dataset):
    default_path = '/home/alienware1/Desktop/Royy/deephar-mas/'
    if dataset == 'Penn_hm':
        return default_path + 'datasets/penn_hm'
    elif dataset == 'UCF':
        return default_path+'datasets/'