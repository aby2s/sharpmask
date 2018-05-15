from collections import deque

from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from tqdm import tqdm
import sys
import os
import argparse


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class RecordCreator(object):
    def __init__(self, data_path, max_file_size=150000):
        self.data_path = data_path
        self.max_file_size = max_file_size

    def create_data(self, target_dir, data_type):
        coco = COCO('{}/annotations/instances_{}.json'.format(self.data_path, data_type))

        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        file_pattern = os.path.join(target_dir, 'coco_part{}.tfrecord')

        tfrecord_id = 0
        imgIds = coco.getImgIds()
        writer = tf.python_io.TFRecordWriter(file_pattern.format(tfrecord_id))

        balance = 0
        total_samples = 0
        pbar = tqdm(total=len(imgIds), desc='Creating record file')

        negative_queue = deque()
        no_anns = 0

        for i, id in enumerate(imgIds):
            img = coco.loadImgs(id)[0]
            if int((i + 1) % self.max_file_size) == 0:
                tfrecord_id += 1
                tfrecord_file = file_pattern.format(tfrecord_id)
                tqdm.write('Creating new tfrecord file id {}, name {}'.format(tfrecord_id, tfrecord_file))
                writer = tf.python_io.TFRecordWriter(tfrecord_file)

            im_path = '{}/images/{}/{}'.format(self.data_path, data_type, img['file_name'])

            annIds = coco.getAnnIds(imgIds=[id], iscrowd=0)
            anns = coco.loadAnns(annIds)
            score = 0
            if len(anns) == 0:
                    no_anns += 1

            for ann in anns:
                score = self.get_score(ann, img)
                if score > 0:
                    mask = Image.new('F', (img['width'], img['height']), color=-1)
                    segs = list(zip(*[iter(ann['segmentation'][0])] * 2))
                    ImageDraw.Draw(mask).polygon(segs, outline=1, fill=1)
                    mask = np.asarray(mask)
                    mask = cv2.resize(mask, (224, 224))
                    mask = np.where(mask == -1.0, -1, 1).astype(np.int8)

                    feature = {'score': _int_feature(score),
                               'image': _bytes_feature(im_path.encode()),
                               'mask': _bytes_feature(mask.tostring())}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    total_samples += 1
                    balance += score
                    break



            if score < 0:
                feature = {'score': _int_feature(score),
                           'image': _bytes_feature(im_path.encode()),
                           'mask': _bytes_feature(mask.tostring())}
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                negative_queue.append(example)

            while balance > 0 and len(negative_queue) > 0:
                example = negative_queue.pop()
                writer.write(example.SerializeToString())
                total_samples += 1
                balance -= 1

            pbar.update()
            pbar.set_description(
                'Creating record file (total samples created={}, balance={})'.format(total_samples, balance))

        print(i)
        tqdm.write('tfrecord file created, total samples {}, balance {}, {} images without annotation'.format(total_samples, balance, no_anns))

    def get_score(self, ann, img):
        ann_ratio = ann['area'] / (img['width'] * img['height'])

        ann_center = (int(ann['bbox'][0] + ann['bbox'][2] / 2), int(ann['bbox'][1] + ann['bbox'][3] / 2))
        ann_center_bounds = (range(int(img['width'] / 4), int(img['width'] - img['width'] / 4)),
                             range(int(img['height'] / 4), int(img['height'] - img['height'] / 4)))
        ann_centered = ann_center[0] in ann_center_bounds[0] and ann_center[1] in ann_center_bounds[1]

        ann_br = (int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3]))
        ann_fully_contained = ann['bbox'][0] > 0 and ann['bbox'][1] > 0 and \
                              ann_br[0] < img['width'] and ann_br[1] < img['height']

        return 1 if  ann['iscrowd'] == 0 and ann_ratio > 0.05 and ann_centered and ann_fully_contained else -1


def main():
    parser = argparse.ArgumentParser(
        description='Use this util to prepare tfrecord files before training DeepMask/SharpMask')

    parser.add_argument('--coco_path', action='store', dest='coco_path',
                        help='A path to downloaded and unzipped coco dataset', required=True)
    parser.add_argument('--train_path', action="store", dest="train_path",
                        help='A path to folder where to put train set tfrecord files', required=True)
    parser.add_argument('--validation_path', action="store", dest="validation_path",
                        help='A path to folder where to put validation set tfrecord files', required=True)
    parser.add_argument('--max_per_file', action="store", dest="max_per_file",
                        type=int, default=70000, help='Max number of samples per single tfrecord file')

    params = parser.parse_args(sys.argv[1:])
    rc = RecordCreator(data_path=params.coco_path)
    print('Preparing validation data')
    rc.create_data(params.validation_path, 'val2017')

    print('Preparing train data')
    rc.create_data(params.train_path, 'train2017')


if __name__ == "__main__":
    sys.exit(main())
