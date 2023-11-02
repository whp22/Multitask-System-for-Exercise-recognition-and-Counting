import os
import sys
import keras.backend as K
if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from utils import *
from utils.pennaction import PennAction
from utils.loader import BatchLoader
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop
from utils.split_model import split_model
from utils.callbacks import SaveModel
from utils.trainer import MultiModelTrainer
from models.multitask_model import multitask
from utils.parallelmodel import ParallelModel
from utils.datasetpath import datasetpath
from utils.tools import ActionEvalCallback
from utils.tools import CountingEvalCallback

'''sys. argv is a list in Python,
 which contains the command-line arguments passed to the script.
  With the len(sys. argv) function you can count the number of arguments.'''

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

default_path = '/home/alienware1/Desktop/Royy/MSEC'

'''build model'''
full_model = multitask()
print("build finish")

'''Load datasets'''
num_predictions = 1
penn_seq = PennAction(datasetpath('Penn_hm'))

count_data_tr = BatchLoader(penn_seq, ['hm'], ['period'],TRAIN_MODE,
        batch_size=8, num_predictions=num_predictions,
        shuffle=True)

action_data_tr = BatchLoader(penn_seq, ['hm'], ['action_id'],TRAIN_MODE,
        batch_size=8, num_predictions=num_predictions,
        shuffle=True)

print('loading data finish')

counting_te = BatchLoader(penn_seq, ['hm'], ['period'], TEST_MODE,
        batch_size=1, shuffle=False)
action_te = BatchLoader(penn_seq, ['hm'], ['action_id'], TEST_MODE,
        batch_size=1, shuffle=False)

'''change keyword here if you wanna train different branch'''
Keyword = 'action'
assert Keyword in ['counting','action'],'Key word is wrong'
if Keyword == 'counting':
    counting_trainable = True
else:
    counting_trainable = False

'''split the whole model to two branches and set trainable layers'''
models = split_model(full_model, counting_trainable = counting_trainable)

'''call back on CPU'''
with tf.device('/cpu:0'):

    counting_callback = CountingEvalCallback(counting_te, eval_model=models[0])
    action_callback = ActionEvalCallback(action_te,eval_model=models[1])

if Keyword == 'counting':
    model_to_save = models[0]
    data_tr = count_data_tr
    loss = mean_squared_error

else:
    model_to_save = models[1]
    data_tr = action_data_tr
    loss = 'categorical_crossentropy'

"""Save model callback."""
save_model = SaveModel(os.path.join(default_path,
                                    'weights/weights{epoch:03d}.hdf5'), model_to_save= model_to_save)

'''loss'''
start_lr = 0.001
optimizer = RMSprop(lr = start_lr)
total_epoch = 10
steps_per_epoch = penn_seq.get_length(TRAIN_MODE)

def end_of_epoch_callback(epoch):
    save_model.on_epoch_end(epoch)
    if Keyword == 'counting':
        score = counting_callback.on_epoch_end(epoch)
    else:
        score = action_callback.on_epoch_end(epoch)

    if epoch in [1, 5]:
        lr = float(K.get_value(optimizer.lr))
        newlr = 0.1 * lr
        K.set_value(optimizer.lr, newlr)
        print('lr_scheduler: lr %s -> %s %d'% (lr, newlr, epoch))

    return score

'''multi-gpu for training'''
GPU_COUNT = 2
'''train counting:'''
model = ParallelModel(model_to_save, GPU_COUNT)

model.compile(optimizer = optimizer, loss = loss)

trainer = MultiModelTrainer(model, data_tr, workers=12,
        print_full_losses=True, logdir = '/home/alienware1/Desktop/Royy/deephar-mas/action_counting/graph')
trainer.train(total_epoch, steps_per_epoch=steps_per_epoch, initial_epoch=0,
        end_of_epoch_callback=end_of_epoch_callback)
