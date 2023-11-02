import os
import json
import numpy as np
from PIL import Image
import mat73


from utils import *
from utils.datasetpath import datasetpath

def load_pennaction_mat_annotation(filename):
    # mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    mat = mat73.loadmat(filename, use_attrdict=True)

    # Respect the order of TEST (0), TRAIN (1). No validation set.
    sequences = [mat['test_seq'],[],[]]
    return sequences


def serialize_index_sequences(sequences):
    frame_idx = []
    for s in range(len(sequences)):
        for f in range(len(sequences[s].frames)):
            frame_idx.append((s, f))

    return frame_idx


def compute_clip_bbox(bbox_dict, seq_idx, frame_list):
    x1 = y1 = np.inf
    x2 = y2 = -np.inf

    for f in frame_list:
        b = bbox_dict['%d.%d' % (seq_idx, f)]
        x1 = min(x1, b[0])
        y1 = min(y1, b[1])
        x2 = max(x2, b[2])
        y2 = max(y2, b[3])

    return np.array([x1, y1, x2, y2])


class PennAction(object):
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.load_annotations(os.path.join(dataset_path, 'new_a7/new_a7.mat'))
        self.speed = [1, 2, 3]
        self.period = [6, 7, 8, 9 ,10, 11, 12, 13, 14]
        self.action_id = [3, 5, 8, 9, 10, 11, 12]
        # self.action_id = [9, 10, 12]

    def load_annotations(self, filename):

        self.sequences = load_pennaction_mat_annotation(filename)
        print('load annotation finish')

    def get_data(self, key, mode):
        """Method to load Penn Action samples specified by mode and key,
        do data augmentation and bounding box cropping.
        """
        output = {}

        seq_idx = key
        seq = self.sequences[mode][seq_idx]
        folder_num = seq.folder_num
        action_id = seq.action_id

        # 16 images in total
        imgt1 = None
        for i in range(16):
            image = 'a%d_hm/%04d/%03d.jpg' % (action_id,folder_num, i)
            # print('go to image_size')
            image = Image.open(os.path.join(self.dataset_path, image))

            imgt = np.asarray(image)
            imgt = np.expand_dims(imgt, axis = -1)

            if i == 0:
                imgt1 = imgt

            else:
                imgt1 = np.concatenate([imgt1, imgt], axis = -1)


        action = np.zeros(self.get_shape('action_id'))
        for i in range(len(self.action_id)):
            if self.action_id[i] == action_id:
                action[i] = 1


        output['hm'] = imgt1
        output['period'] = seq.period
        output['action_id'] = action
        return output

    def get_shape(self, dictkey):

        if dictkey == 'hm':
            return (224, 224, 16)
        if dictkey == 'period':
            # return (len(self.period),)
            return (1,)
        if dictkey == 'action_id':
            return (len(self.action_id),)
        raise Exception('Invalid dictkey ({}) on get_shape!'.format(dictkey))

    def get_length(self, mode):
        return len(self.sequences[mode])



if __name__ == '__main__':

    penn_seq = PennAction(datasetpath('Penn_hm'))
    a = penn_seq.get_data(key=0, mode=TRAIN_MODE)