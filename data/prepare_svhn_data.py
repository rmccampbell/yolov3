#!/usr/bin/env python3

import os
import os.path as osp
import svhn.io
from PIL import Image
try:
    from tqdm import trange
except ImportError:
    trange = range

TRAIN_SIZE = 33402
TEST_SIZE = 13068
EXTRA_SIZE = 202353

if __name__ == '__main__':
    svhn_dir = osp.expanduser('~/.svhn')
    train_dir = osp.join(svhn_dir, 'train')
    test_dir = osp.join(svhn_dir, 'test')
    extra_dir = osp.join(svhn_dir, 'extra')

    dest_dir = os.path.abspath('svhn')
    image_dir = osp.join(dest_dir, 'images')
    label_dir = osp.join(dest_dir, 'labels')

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(osp.join(label_dir, 'train'), exist_ok=True)
    os.makedirs(osp.join(label_dir, 'test'), exist_ok=True)
    os.makedirs(osp.join(label_dir, 'extra'), exist_ok=True)

    try:
        os.symlink(train_dir, osp.join(image_dir, 'train'))
    except:
        pass
    try:
        os.symlink(test_dir, osp.join(image_dir, 'test'))
    except:
        pass
    try:
        os.symlink(extra_dir, osp.join(image_dir, 'extra'))
    except:
        pass

    print('writing train.txt')
    with open(osp.join(dest_dir, 'train.txt'), 'w') as f:
        for i in range(TRAIN_SIZE):
            f.write(osp.join(image_dir, 'train', '{}.png'.format(i+1)) + '\n')

    print('writing test.txt')
    with open(osp.join(dest_dir, 'test.txt'), 'w') as f:
        for i in range(TEST_SIZE):
            f.write(osp.join(image_dir, 'test', '{}.png'.format(i+1)) + '\n')

    print('writing extra.txt')
    with open(osp.join(dest_dir, 'extra.txt'), 'w') as f:
        for i in range(EXTRA_SIZE):
            f.write(osp.join(image_dir, 'extra', '{}.png'.format(i+1)) + '\n')

    s = svhn.io.SVHN()

    if len(os.listdir(osp.join(label_dir, 'train'))) < TRAIN_SIZE:
        print('writing train labels')
        skip = True
        for i in trange(TRAIN_SIZE):
            path = osp.join(label_dir, 'train', '{}.txt'.format(i+1))
            if osp.exists(path) and skip:
                continue
            else:
                skip = False
            with open(path, 'w') as f:
                shape = s.get_full_image(svhn.io.TRAINING, i).shape
                bboxes = s.bounding_boxes(svhn.io.TRAINING, i)
                for bbox in bboxes.boxes:
                    cx = (bbox.left + bbox.width / 2) / shape[1]
                    cy = (bbox.top + bbox.height / 2) / shape[0]
                    w = bbox.width / shape[1]
                    h = bbox.height / shape[0]
                    f.write('{} {} {} {} {}\n'.format(bbox.label%10, cx, cy, w, h))

    if len(os.listdir(osp.join(label_dir, 'test'))) < TEST_SIZE:
        print('writing test labels')
        skip = True
        for i in trange(TEST_SIZE):
            path = osp.join(label_dir, 'test', '{}.txt'.format(i+1))
            if osp.exists(path) and skip:
                continue
            else:
                skip = False
            with open(path, 'w') as f:
                shape = s.get_full_image(svhn.io.TEST, i).shape
                bboxes = s.bounding_boxes(svhn.io.TEST, i)
                for bbox in bboxes.boxes:
                    cx = (bbox.left + bbox.width / 2) / shape[1]
                    cy = (bbox.top + bbox.height / 2) / shape[0]
                    w = bbox.width / shape[1]
                    h = bbox.height / shape[0]
                    f.write('{} {} {} {} {}\n'.format(bbox.label%10, cx, cy, w, h))

    if len(os.listdir(osp.join(label_dir, 'extra'))) < EXTRA_SIZE:
        print('writing extra labels')
        skip = True
        for i in trange(EXTRA_SIZE):
            path = osp.join(label_dir, 'extra', '{}.txt'.format(i+1))
            if osp.exists(path) and skip:
                continue
            else:
                skip = False
            with open(path, 'w') as f:
                shape = s.get_full_image(svhn.io.EXTRA, i).shape
                bboxes = s.bounding_boxes(svhn.io.EXTRA, i)
                for bbox in bboxes.boxes:
                    cx = (bbox.left + bbox.width / 2) / shape[1]
                    cy = (bbox.top + bbox.height / 2) / shape[0]
                    w = bbox.width / shape[1]
                    h = bbox.height / shape[0]
                    f.write('{} {} {} {} {}\n'.format(bbox.label%10, cx, cy, w, h))
