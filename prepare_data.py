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


# _bytes is used for string/char values

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

        for i, id in enumerate(imgIds):
            img = coco.loadImgs(id)[0]
            if int((i + 1) % self.max_file_size) == 0:
                tfrecord_id += 1
                tfrecord_file = file_pattern.format(tfrecord_id)
                tqdm.write('Creating new tfrecord file id {}, name {}'.format(tfrecord_id, tfrecord_file))
                writer = tf.python_io.TFRecordWriter(tfrecord_file)

            im_path = '{}/images/{}/{}'.format(self.data_path, data_type, img['file_name'])
            # I = io.imread('{}/images/{}/{}'.format(self.data_path, self.data_type, img['file_name']))
            # I = cv2.resize(I, (224, 224)).astype(np.float32)
            #
            # if I.shape == (224, 224, 3):
            # I = np.vectorize(lambda x: 256 - x)(I)
            # I[:, :, 0] -= 103.939
            # I[:, :, 1] -= 116.779
            # I[:, :, 2] -= 123.68

            annIds = coco.getAnnIds(imgIds=[id], iscrowd=0)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                mask = Image.new('F', (img['width'], img['height']), color=-1)
                segs = list(zip(*[iter(ann['segmentation'][0])] * 2))
                ImageDraw.Draw(mask).polygon(segs, outline=1, fill=1)
                mask = np.asarray(mask)
                mask = cv2.resize(mask, (224, 224))
                score = self.get_score(mask)
                # mask = cv2.resize(mask, (56, 56))
                mask = np.where(mask == -1.0, -1, 1).astype(np.int8)
                if score > 0:
                    feature = {'score': _int_feature(score),
                               'image': _bytes_feature(im_path.encode()),
                               'mask': _bytes_feature(mask.tostring())}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    total_samples += 1
                    balance += score
                    break

            if score < 0 and balance > 0:
                feature = {'score': _int_feature(score),
                           'image': _bytes_feature(im_path.encode()),
                           'mask': _bytes_feature(mask.tostring())}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                total_samples += 1
                balance += score
            pbar.update()
            pbar.set_description(
                'Creating record file (total samples created={}, balance={})'.format(total_samples, balance))

        tqdm.write('tfrecord file created, total samples {}, balance {}'.format(total_samples, balance))

    # def setupMask(self, mask, length):
    #     for x in range(length):
    #         for y in range(length):
    #             if mask[x][y] != -1.0:
    #                 mask[x][y] = 1.0
    #     return mask

    def get_score(self, mask):
        isCentered = -1
        centerFrame = 16
        offset = int((224 / 2) - centerFrame)
        for x in range(centerFrame * 2):
            for y in range(centerFrame * 2):
                if mask[offset + x][offset + y] == 1:
                    isCentered = 1
                if isCentered == 1:
                    break
            if isCentered == 1:
                break
        isNotTooLarge = 1
        if isCentered == -1:
            return -1
        offset = int((224 - 128) / 2)
        for x in range(128):
            if mask[offset][offset + x] == 1:
                isNotTooLarge = -1
            if mask[offset + x][offset] == 1:
                isNotTooLarge = -1
            if mask[224 - offset][offset + x] == 1:
                isNotTooLarge = -1
            if mask[offset + x][224 - offset] == 1:
                isNotTooLarge = -1
            if isNotTooLarge == -1:
                break
        return isNotTooLarge


def main():
    parser = argparse.ArgumentParser(
        description='Use this util to prepare tfrecord files before training DeepMask/SharpMask')

    parser.add_argument('--coco_path', action='store', dest='coco_path', help='A path to downloaded and unzipped coco dataset', required=True)
    parser.add_argument('--train_path', action="store", dest="train_path", help='A path to folder where to put train set tfrecord files', required=True)
    parser.add_argument('--validation_path', action="store", dest="validation_path", help='A path to folder where to put validation set tfrecord files', required=True)
    parser.add_argument('--max_per_file', action="store", dest="max_per_file", type=int, default=150000, help='Max number of samples per single tfrecord file')

    params = parser.parse_args(sys.argv[1:])
    rc = RecordCreator(data_path=params.coco_path)
    print('Preparing validation data')
    rc.create_data(params.validation_path, 'val2017')

    print('Preparing train data')
    rc.create_data(params.train_path, 'train2017')


if __name__ == "__main__":
    sys.exit(main())
