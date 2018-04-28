import datetime

import gc
import tensorflow as tf
import skimage.io as io
import numpy as np
from tqdm import tqdm

import resnet_model
from im_classes import IM_CLASSES
from PIL import Image, ImageDraw
import cv2
import sys
import glob
import os

IMAGENET_MEANS = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 3], name='img_mean')

def transform_image(image):
    image = tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(image), channels=3, dct_method='INTEGER_ACCURATE'),
                                   size=(224, 224))
    return image - IMAGENET_MEANS


def transform_ds(x):
    keys_to_features = {'score': tf.FixedLenFeature([], tf.int64),
                        'mask': tf.FixedLenFeature([], tf.string),
                        'image': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(x, keys_to_features)
    image = transform_image(parsed_features['image'])
    masks = tf.reshape(tf.decode_raw(parsed_features['mask'], tf.int8), shape=[224, 224])
    return {'score': tf.cast(parsed_features['score'], tf.float32), 'mask': masks, 'image': image}


def validation_ds(x):
    return {'score': tf.zeros((1,), dtype=tf.float32), 'mask': tf.zeros((1, 224, 224), dtype=tf.int8),
            'image': tf.expand_dims(transform_image(x), 0)}
    # return {'score': tf.zeros((1,), dtype=tf.float32), 'mask': tf.zeros((1,224,224), dtype=tf.int8), 'image': tf.expand_dims(x, 0)}


class SharpMask(resnet_model.Model):
    mask_size = 56

    def __init__(self, train_path, validation_path, session=None, resnet_ckpt=None, summary_path=None,
                 checkpoint_path=None, batch_size=32):
        super(SharpMask, self).__init__(
            resnet_size=50,
            bottleneck=True,
            num_classes=1001,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=[3, 4, 6, 3],
            block_strides=[1, 2, 2, 2],
            final_size=2048,
            version=resnet_model.DEFAULT_VERSION,
            data_format=None,
            dtype=resnet_model.DEFAULT_DTYPE
        )
        if session is None:
            self.sess = tf.Session()
        else:
            self.sess = session

        self.train_ds = self._create_dataset(train_path, batch_size)
        self.validation_ds = self._create_dataset(validation_path, batch_size)
        self.iterator = tf.data.Iterator.from_structure(self.train_ds.output_types,
                                                        self.train_ds.output_shapes)

        self.training_init_op = self.iterator.make_initializer(self.train_ds)
        self.validation_init_op = self.iterator.make_initializer(self.validation_ds)

        self.it_next = self.iterator.get_next()


        if summary_path is not None:
            self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        self.checkpoint_file = os.path.join(checkpoint_path, 'sharpmask.ckpt')

        self.resnet_output = self(self.it_next['image'], False)

        self.saver = tf.train.Saver()

        if resnet_ckpt is not None:
            self.saver.restore(self.sess, resnet_ckpt)

        self.block_layers = [self.sess.graph.get_tensor_by_name("resnet_model/block_layer{}:0".format(i + 1)) for i in
                             range(4)]

        self.training_mode = tf.placeholder_with_default(True, shape=())

        #trunk = self.block_layers[-1]

        with tf.variable_scope("deepmask_trunk"):
            #trunk = tf.layers.conv2d(self.block_layers[-1], 2048, (1, 1), activation=tf.nn.relu)
            trunk = tf.layers.conv2d(self.block_layers[-1], 512, (1, 1), activation=tf.nn.relu, data_format=self.data_format)
            trunk = tf.layers.flatten(trunk)
            trunk = tf.layers.dense(trunk, 512)
        self.sess.run(tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='deepmask_trunk')))

        with tf.variable_scope("segmentation_branch"):
            seg_predictions = tf.layers.dense(trunk, self.mask_size * self.mask_size)
            seg_predictions = tf.reshape(seg_predictions, [-1, self.mask_size, self.mask_size, 1])
            self.seg_predictions = tf.squeeze(tf.image.resize_bilinear(seg_predictions, [224, 224]), 3)

            # seg_predictions = tf.layers.conv2d(trunk, 512, (1, 1), activation=tf.nn.relu, data_format=self.data_format)
            # seg_predictions = tf.layers.flatten(seg_predictions)
            # seg_predictions = tf.layers.dense(seg_predictions, 512)
            # seg_predictions = tf.layers.dense(seg_predictions, self.mask_size * self.mask_size)
            # seg_predictions = tf.reshape(seg_predictions, [-1, self.mask_size, self.mask_size, 1])
            # self.seg_predictions = tf.squeeze(tf.image.resize_bilinear(seg_predictions, [224, 224]), 3)

        self.sess.run(
            tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='segmentation_branch')))

        with tf.variable_scope("score_branch"):
            score_predictions = tf.layers.dropout(trunk, rate=0.5, training=self.training_mode)
            score_predictions = tf.layers.dense(score_predictions, 1024, activation=tf.nn.relu)
            score_predictions = tf.layers.dropout(score_predictions, rate=0.5, training=self.training_mode)
            self.score_predictions = tf.layers.dense(score_predictions, 1, name='score_out')

            # score_predictions = tf.layers.max_pooling2d(trunk, padding='SAME', pool_size=(2, 2), strides=(2, 2))
            # score_predictions = tf.layers.flatten(score_predictions)
            # score_predictions = tf.layers.dense(score_predictions, 512, activation=tf.nn.relu)
            # score_predictions = tf.layers.dropout(score_predictions, rate=0.5, training=self.training_mode)
            #
            # score_predictions = tf.layers.dense(score_predictions, 1024, activation=tf.nn.relu)
            # score_predictions = tf.layers.dropout(score_predictions, rate=0.5, training=self.training_mode)
            # self.score_predictions = tf.layers.dense(score_predictions, 1, name='score_out')

        self.sess.run(
            tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='score_branch')))

        with tf.variable_scope("refinement"):
            channel_axis = 1 if self.data_format == "channels_first" else 3
            #height_axis, width_axis = (2, 3) if self.data_format == "channels_first" else (1, 2)
            M = None
            for i in range(4, 0, -1):
                F = self.block_layers[i - 1]
                S = tf.layers.conv2d(F, int(int(F.shape[channel_axis]) / 2), (3, 3), padding='SAME',
                                     activation=tf.nn.relu, data_format=self.data_format,
                                     name='horizontal_1_{}'.format(i))
                M = tf.concat((M, S), axis=channel_axis) if M is not None else S
                M = tf.layers.conv2d(M, S.shape[channel_axis], (3, 3), padding='SAME', activation=tf.nn.relu,
                                     data_format=self.data_format, name='horizontal_2_{}'.format(i))
                if self.data_format == "channels_first":
                    M = tf.transpose(M, perm=[0, 2, 3, 1])
                    M = tf.image.resize_bilinear(M, [M.shape[1] * 2, M.shape[2] * 2])
                    M = tf.transpose(M, perm=[0, 3, 1, 2])
                else:
                    M = tf.image.resize_bilinear(M, [M.shape[1] * 2, M.shape[2] * 2])

            refinement_out = tf.layers.conv2d(M, 1, (3, 3), padding='SAME', activation=tf.nn.relu,
                                              data_format=self.data_format, name='final_refinement')
            if self.data_format == "channels_first":
                refinement_out = tf.transpose(refinement_out, perm=[0, 2, 3, 1])

            refinement_out = tf.image.resize_bilinear(refinement_out, [refinement_out.shape[1] * 2, refinement_out.shape[2] * 2])
            self.refinement_prediction = tf.squeeze(refinement_out, axis=3)

        self.sess.run(tf.initialize_variables(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='refinement')))

        with tf.variable_scope("metrics"):
            score_metric_prediction = tf.where(self.score_predictions > 0.0,
                                               tf.ones_like(self.score_predictions),
                                               -tf.ones_like(self.score_predictions))
            self.score_accuracy_metric, self.score_accuracy_update = tf.metrics.accuracy(self.it_next['score'],
                                                                                         score_metric_prediction)

            mask_indices = tf.where(self.it_next['score'] > 0)
            seg_metric_prediction = tf.gather(self.seg_predictions, mask_indices)
            seg_metric_prediction = tf.where(seg_metric_prediction > 0.0, tf.ones_like(seg_metric_prediction),
                                             tf.zeros_like(seg_metric_prediction))
            seg_mask = tf.gather(self.it_next['mask'], mask_indices)
            seg_mask = tf.where(seg_mask > 0, tf.ones_like(seg_mask), tf.zeros_like(seg_mask))
            self.seg_iou_metric, self.seg_iou_update = tf.metrics.mean_iou(seg_mask, seg_metric_prediction, 2)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_file)

    def _create_dataset(self, data_path, batch_size):
        tfrecord_files = glob.glob(os.path.join(data_path, '*.tfrecord'))
        dataset = tf.data.TFRecordDataset(tfrecord_files, buffer_size=1572864000)
        dataset = dataset.shuffle(20000)
        dataset = dataset.map(transform_ds, num_parallel_calls=20)
        dataset = dataset.batch(32)

        return dataset

    def binary_regression_loss(self, score_factor=1.0 / 32):
        score_target = self.it_next['score']
        mask_target = tf.cast(self.it_next['mask'], tf.float32)
        segmentation_loss = tf.reduce_mean(
            (1.0 + score_target) / 2.0 * tf.reduce_mean(tf.log(1.0 + tf.exp(-self.seg_predictions * mask_target)),
                                                        axis=[1, 2]))
        score_loss = tf.reduce_mean(tf.log(1.0 + tf.exp(-score_target * self.score_predictions))) * score_factor
        return score_loss, segmentation_loss

    def fit_sharpmask(self, epochs=300, lr=0.001, weight_decay=0.00005):
        with tf.variable_scope("sharpmask_training"):
            _, segmentation_loss = self.binary_regression_loss()

            global_step = tf.Variable(initial_value=0)
            lr_var = lr / (1 + weight_decay * global_step)
            segmentation_opt = tf.train.MomentumOptimizer(learning_rate=lr_var, momentum=0.9, use_nesterov=True)
            segmentation_opt_op = segmentation_opt.minimize(segmentation_loss, global_step=global_step,
                                                            var_list=tf.get_collection(
                                                                key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                                scope='score_branch'))

        self.sess.run(
            tf.initialize_variables(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='sharpmask_training')))

        self._fit_cycle(epochs,
                        progress_ops_dict={'segmentation_loss': segmentation_loss,
                                           'segmentation_iou': self.seg_iou_metric},
                        opt_ops=[segmentation_opt_op],
                        metric_update_ops=[self.seg_iou_update])

        print('Sharp mask fit cycle completed')

    def run_validation(self, progress_ops_dict, metric_update_ops, validation_steps_count=None):
        if progress_ops_dict is not None:
            progress_ops_names, progress_ops = zip(*progress_ops_dict.items())
            progress_ops = list(progress_ops)
        else:
            progress_ops_names = ['segmentation_iou', 'score_accuracy']
            progress_ops = [self.seg_iou_metric, self.score_accuracy_metric]
            metric_update_ops = [self.seg_iou_update, self.score_accuracy_update]

        validation_ops = metric_update_ops + progress_ops

        pbar = tqdm(total=validation_steps_count, desc='Validation', file=sys.stdout)
        counter = 0

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.validation_init_op)

        while True:
            try:
                # res = self.sess.run([score_loss, segmentation_loss, self.score_accuracy_update, self.seg_iou_update, self.score_accuracy_metric, self.seg_iou_metric], feed_dict={self.training_mode: False})
                progress = self.sess.run(validation_ops, feed_dict={self.training_mode: False})[-len(progress_ops):]
                counter += 1
                pbar.update()
                pbar.set_description('Validation ({})'.format(
                    ', '.join(['{}={}'.format(name, val) for name, val in zip(progress_ops_names, progress)])))
            except tf.errors.OutOfRangeError as oe:
                break

        result = {name: value for name, value in zip(progress_ops_names, progress)}
        result['total_steps'] = counter

        return result

    def _fit_cycle(self, epochs, progress_ops_dict, opt_ops, metric_update_ops):
        progress_ops_names, progress_ops = zip(*progress_ops_dict.items())
        training_ops = opt_ops + metric_update_ops + list(progress_ops)

        train_steps_per_epoch = None
        validation_steps_per_epoch = None

        for e in range(epochs):
            tic = datetime.datetime.now()
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(self.training_init_op)

            print()
            tqdm.write("----- Epoch {}/{} ;  -----".format(e + 1, epochs))
            pbar = tqdm(total=train_steps_per_epoch, desc='Training', file=sys.stdout)
            train_steps_per_epoch = 0

            while True:
                try:
                    # res = self.sess.run([score_loss, segmentation_loss, opt_op, self.score_accuracy_update, self.seg_iou_update, self.score_accuracy_metric, self.seg_iou_metric])
                    progress = self.sess.run(training_ops)[-len(progress_ops):]
                    pbar.update()
                    pbar.set_description('Training ({})'.format(
                        ', '.join(['{}={}'.format(name, val) for name, val in zip(progress_ops_names, progress)])))
                    train_steps_per_epoch += 1
                except tf.errors.OutOfRangeError as oe:
                    break

            del pbar
            validation_results = self.run_validation(progress_ops_dict, metric_update_ops, validation_steps_per_epoch)
            training_report = ', '.join(
                ['Training {}={}'.format(name, val) for name, val in zip(progress_ops_names, progress)])
            validation_report = ', '.join(['Validation {}={}'.format(name, val) for name, val in validation_results.items()])
            validation_steps_per_epoch = validation_results['total_steps']
            self.saver.save(self.sess, self.checkpoint_file)
            gc.collect()
            toc = datetime.datetime.now()
            tqdm.write(
                "----- Epoch {} finished in {} -- {}. {}".format(e, toc - tic, training_report, validation_report))

    def fit_deepmask(self, epochs=300, lr=0.001, score_factor=1.0 / 32, weight_decay=0.00005):
        with tf.variable_scope("deepmask_training"):
            score_loss, segmentation_loss = self.binary_regression_loss()

            global_step = tf.Variable(initial_value=0.0)
            lr_var = lr / (1.0 + weight_decay * global_step)
            segmentation_opt = tf.train.MomentumOptimizer(learning_rate=lr_var, momentum=0.9, use_nesterov=True)
            segmentation_opt_op = segmentation_opt.minimize(segmentation_loss, global_step=global_step)

            score_opt = tf.train.MomentumOptimizer(learning_rate=lr_var, momentum=0.9, use_nesterov=True)
            score_gvs = score_opt.compute_gradients(score_loss, tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                  scope='score_branch'))
            score_opt_op = score_opt.apply_gradients(
                score_gvs)  # [(tf.clip_by_value(g, -5.0, 5.0) if g is not None else g, v) for g,v in gvs])

        self.sess.run(
            tf.initialize_variables(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='deepmask_training')))

        self._fit_cycle(epochs,
                        progress_ops_dict={'segmentation_loss': segmentation_loss, 'score_loss': score_loss,
                                           'segmentation_iou': self.seg_iou_metric,
                                           'score_accuracy': self.score_accuracy_metric},
                        opt_ops=[segmentation_opt_op, score_opt_op],
                        metric_update_ops=[self.seg_iou_update, self.score_accuracy_update])

        print('Deep mask fit cycle completed')

    def eval(self, image_path):
        image = io.imread(image_path)
        image1 = cv2.resize(image, (224, 224)).astype(np.float32)
        eval_dataset = tf.data.Dataset.from_tensor_slices(np.array([image_path]))
        eval_dataset = eval_dataset.map(validation_ds)
        init = self.iterator.make_initializer(eval_dataset)
        self.sess.run([init])
        score_predictions, seg_predictions = self.sess.run([self.score_predictions, self.seg_predictions])
        seg_predictions = seg_predictions[0]
        mask = np.where(seg_predictions > -1, 255, 0)
        mask = np.expand_dims(mask, axis=2).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = Image.fromarray(mask)
        image = Image.fromarray(image)
        mask = mask.convert('RGBA')
        image = image.convert('RGBA')
        new_img = Image.blend(image, mask, 0.5)

        new_img.save("file.png")
        print(seg_predictions)

    def eval_resnet(self, image_path):
        eval_dataset = tf.data.Dataset.from_tensor_slices(np.array([image_path]))
        eval_dataset = eval_dataset.map(validation_ds)
        init = self.iterator.make_initializer(eval_dataset)
        self.sess.run([init])
        prediction = self.sess.run([self.resnet_output])
        print(IM_CLASSES[np.argmax(prediction[0])])

    def eval1(self, image_path):
        image = io.imread(image_path)
        image = cv2.resize(image, (224, 224)).astype(np.float32)
        eval_dataset = tf.data.Dataset.from_tensor_slices(np.array([image]))
        eval_dataset = eval_dataset.map(validation_ds)
        init = self.iterator.make_initializer(eval_dataset)
        self.sess.run([init])
        res = self.sess.run([self.resnet_output])
        print(IM_CLASSES[np.argmax(res[0])])


def main(eval=False):
    if eval:
        # dm = SharpMask(train_path='D:/data/coco/tfrecord', validation_path='D:/data/coco/tfrecord_val',
        #                resnet_ckpt="D:\\data\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary", checkpoint_path='D:/data/coco/sm_model')
        dm = SharpMask(train_path='D:/data/coco/tfrecord', validation_path='D:/data/coco/tfrecord_val',
                       resnet_ckpt="D:\\data\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary",
                       checkpoint_path='D:/data/coco/sm_model')
        # dm.restore()
        # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000001584.jpg')
        #dm.eval_resnet('bear.jpg')
        # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000000785.jpg')
        # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000000285.jpg')
        # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000007784.jpg')

        # dm.eval('nig.jpg')
    else:
        dm = SharpMask(train_path='E:/data/ml/coco/tfrecord_224', validation_path='E:/data/ml/coco/tfrecord_val_224',
                       resnet_ckpt="E:\\data\\ml\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary",
                       checkpoint_path="E:/data/ml/coco/sm_dump")
        # dm = SharpMask(train_path='D:/data/coco/tfrecord_val_224', validation_path='D:/data/coco/tfrecord_val_224',
        #               resnet_ckpt="D:\\data\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary", checkpoint_path="D:/data/coco/sm_dump")
        # dm.restore()
        #dm.eval_resnet('test_images/bear.jpg')

        dm.fit_deepmask()


if __name__ == "__main__":
    sys.exit(main())
