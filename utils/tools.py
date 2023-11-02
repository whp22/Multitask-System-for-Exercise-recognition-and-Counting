import os

import numpy as np
import json
import time

from keras.callbacks import Callback

from .loader import BatchLoader


def eval_action(model, datagen, verbose=1, logdir=None):
    '''for eval action for confusion matrix'''

    num_blocks = len(model.outputs)
    num_samples = len(datagen)
    start = time.time()

    for i in range(num_samples):
        [x], [y] = datagen[i]
        if 'y_true' not in locals():
            y_true = np.zeros((num_samples,) + y.shape[1:])
            y_pred = np.zeros((num_samples,) + y.shape[1:])

        y_true[i, :] = y
        pred = model.predict([x])
        # for b in range(num_blocks):
        y_pred[i, :] = pred

    dt = time.time() - start


    # for b in range(num_blocks):
    '''correct period accuracy'''
    correct = np.equal(np.argmax(y_true, axis=-1),
            np.argmax(y_pred[:, :], axis=-1), dtype=np.float)
    scores = sum(correct) / len(correct)
    # num_pullup = sum(np.argmax(y_pred[:, :], axis=-1) == 0)
    # num_pushup = sum(np.argmax(y_pred[:, :], axis=-1) == 1)
    # num_squat = sum(np.argmax(y_pred[:, :], axis=-1) == 2)
    # print('len_pred_pullup:', num_pullup)
    # print('len_pred_pushup:', num_pushup)
    # print('len_pred_squat:', num_squat)
    num_benchpress = sum(np.argmax(y_pred[:, :], axis=-1) == 0)
    num_cleanjerk = sum(np.argmax(y_pred[:, :], axis=-1) == 1)
    num_jumpingjack = sum(np.argmax(y_pred[:, :], axis=-1) == 2)
    num_pullup = sum(np.argmax(y_pred[:, :], axis=-1) == 3)
    num_pushup = sum(np.argmax(y_pred[:, :], axis=-1) == 4)
    num_situp = sum(np.argmax(y_pred[:, :], axis=-1) == 5)
    num_squat = sum(np.argmax(y_pred[:, :], axis=-1) == 6)


    print('len_samples:', num_samples)
    print('len_pred_benchpress:', num_benchpress)
    print('len_pred_cleanjerk:', num_cleanjerk)
    print('len_pred_jumpingjack:', num_jumpingjack)
    print('len_pred_pullup:', num_pullup)
    print('len_pred_pushup:', num_pushup)
    print('len_pred_situp:', num_situp)
    print('len_pred_squat:', num_squat)



    if verbose:
        print('PennAction correct action recognition acc %.2f:' % (100*scores))

    if verbose:
        print('\n%d samples in %.1f sec: %.1f clips per sec' \
                % (num_samples, dt, num_samples / dt))

    #
    # if verbose:
    #     print('\n%d samples in %.1f sec: %.1f clips per sec' \
    #             % (num_samples, dt, num_samples / dt))

    return scores


def eval_counting(model, datagen, verbose=1, logdir=None):

    num_blocks = len(model.outputs)
    num_samples = len(datagen)
    start = time.time()

    for i in range(num_samples):
        [x], [y] = datagen[i]
        if 'y_true' not in locals():
            y_true = np.zeros((num_samples,) + y.shape[1:])
            y_pred = np.zeros((num_samples,) + y.shape[1:])

        y_true[i, :] = y
        pred = model.predict([x])
        # for b in range(num_blocks):
        y_pred[i, :] = pred

    dt = time.time() - start


    # for b in range(num_blocks):
    '''correct period accuracy'''
    # correct = np.equal(y_true, y_pred, dtype=np.float)
    # scores = sum(correct) / len(correct)
    '''MAE error'''
    a1 = abs(y_true - y_pred)
    mae_abs = a1/y_true
    scores_mae = np.mean(mae_abs)
    scores_accuray = np.mean(a1)

    stand_de = np.square(y_true - y_pred)
    stand_de = np.sqrt(np.mean(stand_de))

    '''OBO error'''
    correct_obo = abs(y_true - y_pred) <= 1
    scores_obo = sum(correct_obo)/len(correct_obo)

    if verbose:
        # print('PennAction correct counting acc %.3f:' % scores)
        print('PennAction MAE acc %.3f:' % scores_mae)
        print('Pennaction standard deviation is: %.3f' % stand_de)
        print('PennAction OBO acc %.3f:' % scores_obo)
        print('PennAction accuracy acc %.3f:' % scores_accuray)

    if verbose:
        print('\n%d samples in %.1f sec: %.1f clips per sec' \
                % (num_samples, dt, num_samples / dt))

    #
    # if verbose:
    #     print('\n%d samples in %.1f sec: %.1f clips per sec' \
    #             % (num_samples, dt, num_samples / dt))

    return scores_mae, stand_de, scores_obo

class ActionEvalCallback(Callback):

    def __init__(self, data, batch_size=1, eval_model=None,
            logdir=None):

        self.data = data
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        if type(self.data) == BatchLoader:
            # scores, scores_mae, scores_obo = eval_singleclip_gt_bbox_generator(model, self.data)
            scores = eval_action(model, self.data)

        epoch += 1

        cur_best = scores
        self.scores[epoch] = cur_best

        print('Best action score in PennAction is %.1f at epoch %d' % \
              (100 * self.best_score, self.best_epoch))

        return 100 * scores


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the maximum value from a dict
            return max(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the maximum value from a dict
            return self.scores[self.best_epoch]
        else:
            return 0


class CountingEvalCallback(Callback):

    def __init__(self, data, batch_size=1, eval_model=None,
            logdir=None):

        self.data = data
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        if type(self.data) == BatchLoader:
            # scores, scores_mae, scores_obo = eval_singleclip_gt_bbox_generator(model, self.data)
            scores_mae, stand_de, scores_obo = eval_counting(model, self.data)

        else:
            scores_mae = 0
            scores_obo = 0

        epoch += 1
        # if self.logdir is not None:
        #     if not hasattr(self, 'logarray'):
        #         self.logarray = {}
        #     self.logarray[epoch] = scores
        #     with open(os.path.join(self.logdir, 'penn_val.json'), 'w') as f:
        #         json.dump(self.logarray, f)

        cur_best = scores_mae
        self.scores[epoch] = cur_best

        print('Best MAE score in PennAction is %.1f at epoch %d' % \
              (100 * self.best_score, self.best_epoch))

        return 100 * scores_mae

    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the maximum value from a dict
            return max(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the maximum value from a dict
            return self.scores[self.best_epoch]
        else:
            return 0

# Aliases.
# eval_singleclip = eval_singleclip_gt_bbox
# eval_singleclip_generator = eval_singleclip_gt_bbox_generator