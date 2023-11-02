import os
import sys
"""For calculate confusion matrix"""
if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# import msec
from utils import *

from utils.tools import eval_counting
from utils.pennaction import PennAction
from utils.loader import BatchLoader
from utils.split_model import split_model
from models.multitask_model import multitask
from utils.datasetpath import datasetpath
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
default_path = '/home/alienware1/Desktop/Royy/deephar-mas/'
'''build model'''
full_model = multitask()
print("build finish")

"""Load datasets"""
num_predictions = 1
penn_seq = PennAction(datasetpath('Penn_hm'))

'''load weights'''
full_model.load_weights(default_path+'action_counting/train24_c7/aweights/weights009.hdf5',by_name=True)
print('load weight finished')

action_te = BatchLoader(penn_seq, ['hm'], ['action_id'], TEST_MODE,
        batch_size=1, shuffle=False)
counting_te = BatchLoader(penn_seq, ['hm'], ['period'], TEST_MODE,
        batch_size=1, shuffle=False)

Keyword = 'counting'
assert Keyword in ['counting','action'],'Key word is wrong'
if Keyword == 'counting':
    counting_trainable = True
else:
    counting_trainable = False

models = split_model(full_model, counting_trainable = counting_trainable)

'''action recognition evaluation and counting evaluation'''
# action_callback = eval_action(model = models[1], datagen=action_te)
counting_callback = eval_counting(model = models[0], datagen=counting_te)