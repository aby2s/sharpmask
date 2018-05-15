import datetime

import gc
import tensorflow as tf
import skimage.io as io
import numpy as np
from tqdm import tqdm
import itertools
import resnet_model
from im_classes import IM_CLASSES
from PIL import Image
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


class SharpMask(resnet_model.Model):
    mask_size = 224
    types = {'score': tf.float32, 'mask': tf.int8, 'image': tf.float32}
    shapes = {'score': tf.TensorShape([None]),
              'mask': tf.TensorShape([None, mask_size, mask_size]),
              'image': tf.TensorShape([None, mask_size, mask_size, 3])}

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

        it_structure = tf.data.Iterator.from_structure(self.types, self.shapes)
        self.iterator = it_structure.get_next()

        self.image_placeholder = tf.placeholder_with_default("", shape=())

        self.image_input = self.iterator['image']
        self.score_target = self.iterator['score']
        self.seg_target = self.iterator['mask']

        self.score_placeholder = tf.placeholder_with_default([1.0], (1,))
        self.mask_placeholder = tf.placeholder_with_default(tf.ones((1, self.mask_size, self.mask_size), dtype=tf.int8),
                                                            (1, self.mask_size, self.mask_size))
        dummy_ds = tf.data.Dataset.from_tensor_slices(
            {'image': tf.expand_dims(transform_image(self.image_placeholder), 0),
             'score': self.score_placeholder, 'mask': self.mask_placeholder}).map(
            lambda x: {'score': tf.expand_dims(x['score'], 0), 'mask': tf.expand_dims(x['mask'], 0),
                       'image': tf.expand_dims(x['image'], 0)})
        self.placeholder_init_op = it_structure.make_initializer(dummy_ds)

        if train_path is not None:
            self.train_ds = self._create_dataset(train_path, batch_size)
            self.training_init_op = it_structure.make_initializer(self.train_ds)

        if validation_path is not None:
            self.validation_ds = self._create_dataset(validation_path, batch_size)
            self.validation_init_op = it_structure.make_initializer(self.validation_ds)

        if summary_path is not None:
            self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        self.checkpoint_file = os.path.join(checkpoint_path, 'sharpmask.ckpt')

        self.resnet_output = self(self.image_input, False)

        if resnet_ckpt is not None:
            saver = tf.train.Saver()
            saver.restore(self.sess, resnet_ckpt)

        self.block_layers = [self.sess.graph.get_tensor_by_name("resnet_model/block_layer{}:0".format(i + 1)) for i in
                             range(4)]

        self.training_mode = tf.placeholder_with_default(True, shape=())

        with tf.variable_scope("deepmask_trunk"):
            trunk = tf.layers.conv2d(self.block_layers[-1], 512, (1, 1), activation=tf.nn.relu,
                                     data_format=self.data_format)
            trunk = tf.layers.flatten(trunk)
            trunk = tf.layers.dense(trunk, 512)
        self.sess.run(
            tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='deepmask_trunk')))

        with tf.variable_scope("segmentation_branch"):
            seg_predictions = tf.layers.dense(trunk, 56 * 56)
            seg_predictions = tf.reshape(seg_predictions, [-1, 56, 56, 1])
            self.dm_seg_prediction = tf.squeeze(tf.image.resize_bilinear(seg_predictions, [224, 224]), 3)

        self.sess.run(
            tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='segmentation_branch')))

        with tf.variable_scope("score_branch"):
            score_predictions = tf.layers.dropout(trunk, rate=0.5, training=self.training_mode)
            score_predictions = tf.layers.dense(score_predictions, 1024, activation=tf.nn.relu)
            score_predictions = tf.layers.dropout(score_predictions, rate=0.5, training=self.training_mode)
            self.score_predictions = tf.layers.dense(score_predictions, 1, name='score_out')

        self.sess.run(
            tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='score_branch')))

        #self.saver = tf.train.Saver()

        k = 32
        with tf.variable_scope("refinement"):
            M = tf.layers.dense(trunk, k*7*7, name='vertical_0')
            M = tf.reshape(M, [-1, k, 7, 7]) if self.data_format == "channels_first" else tf.reshape(M, [-1, 7, 7, k])

            for i in range(1, 5):
                ki = int(k/2**(i-1))
                knext = int(ki/2)

                F = self.block_layers[4-i]
                S = tf.layers.conv2d(F, 64 if i < 4 else 32, (3, 3), padding='SAME',
                                     activation=tf.nn.relu, data_format=self.data_format,
                                     name='horizontal_{}_64'.format(i))

                S = tf.layers.conv2d(S, ki, (3, 3), padding='SAME',
                                     activation=tf.nn.relu, data_format=self.data_format,
                                     name='horizontal_{}_{}'.format(i, ki))
                S = tf.layers.conv2d(S, knext, (3, 3), padding='SAME', data_format=self.data_format,
                                     name='horizontal_{}_{}'.format(i, knext))

                M = tf.layers.conv2d(M, k/2**(i-1), (3, 3), padding='SAME',
                                     activation=tf.nn.relu, data_format=self.data_format,
                                     name='vertical_{}_{}'.format(i, ki))
                M = tf.layers.conv2d(M, knext, (3, 3), padding='SAME', data_format=self.data_format,
                                     name='vertical_{}_{}'.format(i, knext))

                M = tf.nn.relu(S + M)
                if self.data_format == "channels_first":
                    M = tf.transpose(M, perm=[0, 2, 3, 1])
                    M = tf.image.resize_bilinear(M, [M.shape[1] * 2, M.shape[2] * 2])
                    M = tf.transpose(M, perm=[0, 3, 1, 2])
                else:
                    M = tf.image.resize_bilinear(M, [M.shape[1] * 2, M.shape[2] * 2])

            refinement_out = tf.layers.conv2d(M, 1, (3, 3), padding='SAME',
                                              data_format=self.data_format, name='refinement_out')
            if self.data_format == "channels_first":
                refinement_out = tf.transpose(refinement_out, perm=[0, 2, 3, 1])

            refinement_out = tf.image.resize_bilinear(refinement_out,
                                                      [refinement_out.shape[1] * 2, refinement_out.shape[2] * 2])
            refinement_out = tf.squeeze(refinement_out, axis=3)
            self.refinement_prediction = refinement_out


        self.sess.run(
            tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='refinement')))

        with tf.variable_scope("metrics"):
            score_metric_prediction = tf.where(self.score_predictions > 0.0,
                                               tf.ones_like(self.score_predictions),
                                               -tf.ones_like(self.score_predictions))
            self.score_accuracy_metric, self.score_accuracy_update = tf.metrics.accuracy(self.score_target,
                                                                                         score_metric_prediction)

            self.dm_seg_iou_metric, self.dm_seg_iou_update = self._create_seg_metrics(self.dm_seg_prediction)
            self.sm_seg_iou_metric, self.sm_seg_iou_update = self._create_seg_metrics(
                self.refinement_prediction)

        self.saver = tf.train.Saver()

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_file)

    def fit_deepmask(self, epochs=25, lr=0.001, score_factor=1.0 / 32, weight_decay=0.00005):
        with tf.variable_scope("deepmask_training"):
            score_loss, segmentation_loss = self._binary_regression_loss(self.dm_seg_prediction,
                                                                         score_factor=score_factor)

            lr_var = tf.constant(lr)  # tf.train.inverse_time_decay(lr, global_step, 1,weight_decay)
            weight_loss, weight_vars = self._weight_decay()
            weight_decay_opt = tf.train.GradientDescentOptimizer(learning_rate=weight_decay)
            weight_decay_opt_op = weight_decay_opt.minimize(weight_loss, var_list=weight_vars)
            opt = tf.train.MomentumOptimizer(learning_rate=lr_var, momentum=0.9, use_nesterov=True)
            opt_op = opt.minimize(segmentation_loss+score_loss)

        self.sess.run(
            tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='deepmask_training')))

        self._fit_cycle(epochs, lr_var,
                        progress_ops_dict={'segmentation_loss': segmentation_loss, 'score_loss': score_loss,
                                           'segmentation_iou': self.dm_seg_iou_metric,
                                           'score_accuracy': self.score_accuracy_metric},
                        opt_ops=[opt_op, weight_decay_opt_op],
                        metric_update_ops=[self.dm_seg_iou_update, self.score_accuracy_update])

        print('Deep mask fit cycle completed')

    def fit_sharpmask(self, epochs=25, lr=0.001, weight_decay=0.00005):
        with tf.variable_scope("sharpmask_training"):
            _, segmentation_loss = self._binary_regression_loss(self.refinement_prediction)

            global_step = tf.Variable(initial_value=0)
            lr_var = tf.constant(lr)

            segmentation_opt = tf.train.MomentumOptimizer(learning_rate=lr_var, momentum=0.9, use_nesterov=True)
            segmentation_opt_op = segmentation_opt.minimize(segmentation_loss, global_step=global_step,
                                                            var_list=tf.get_collection(
                                                                key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                                scope='refinement'))

        self.sess.run(
            tf.variables_initializer(tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='sharpmask_training')))

        self._fit_cycle(epochs, lr_var,
                        progress_ops_dict={'segmentation_loss': segmentation_loss,
                                           'segmentation_iou': self.sm_seg_iou_metric},
                        opt_ops=[segmentation_opt_op],
                        metric_update_ops=[self.sm_seg_iou_update])

        print('Sharp mask fit cycle completed')

    def deepmask_validation(self):
        self._run_validation({'segmentation_iou': self.dm_seg_iou_metric, 'score_accuracy': self.score_accuracy_metric},
                             metric_update_ops=[self.dm_seg_iou_update, self.score_accuracy_update])

    def sharpmask_validation(self):
        self._run_validation({'segmentation_iou': self.sm_seg_iou_metric},
                             metric_update_ops=[self.sm_seg_iou_update])

    def eval_sharpmask(self, eval_source, eval_target):
        self._eval_prediction(eval_source, eval_target, self.refinement_prediction)

    def eval_deepmask(self, eval_source, eval_target):
        self._eval_prediction(eval_source, eval_target, self.dm_seg_prediction)

    def _create_seg_metrics(self, seg_predictions):
        mask_indices = tf.where(self.score_target > 0)
        seg_metric_prediction = tf.gather(seg_predictions, mask_indices)
        seg_metric_prediction = tf.where(seg_metric_prediction > 0.0, tf.ones_like(seg_metric_prediction),
                                         tf.zeros_like(seg_metric_prediction))
        seg_mask = tf.gather(self.seg_target, mask_indices)
        seg_mask = tf.where(seg_mask > 0, tf.ones_like(seg_mask), tf.zeros_like(seg_mask))
        return tf.metrics.mean_iou(seg_mask, seg_metric_prediction, 2)

    def _eval_prediction(self, eval_source, eval_target, seg_predictions, threshold=-1.0):
        self.sess.run([self.placeholder_init_op],
                      feed_dict={self.image_placeholder: eval_source, self.training_mode: False})
        score_predictions, seg_predictions = self.sess.run([self.score_predictions, seg_predictions])

        print('Predicted score is {}'.format(score_predictions[0]))

        eval_image = io.imread(eval_source)
        mask = np.where(seg_predictions[0] > threshold, 255, 0)
        mask = np.expand_dims(mask, axis=2).astype(np.uint8)
        mask = cv2.resize(mask, (eval_image.shape[1], eval_image.shape[0]))
        mask = Image.fromarray(mask)
        mask = mask.convert('RGB')

        eval_image = Image.fromarray(eval_image)
        eval_image = eval_image.convert('RGB')

        target_img = Image.blend(eval_image, mask, 0.5)
        target_img.save(eval_target)

        print('Image with the mask applied stored at {}'.format(eval_target))

    def _eval_resnet(self, eval_source):
        self.sess.run([self.placeholder_init_op], feed_dict={self.image_placeholder: eval_source})
        prediction = self.sess.run([self.resnet_output])
        return IM_CLASSES[np.argmax(prediction[0])]

    def _create_dataset(self, data_path, batch_size):
        tfrecord_files = glob.glob(os.path.join(data_path, '*.tfrecord'))
        dataset = tf.data.TFRecordDataset(tfrecord_files, buffer_size=1572864000)
        dataset = dataset.shuffle(20000)
        dataset = dataset.map(transform_ds, num_parallel_calls=20)
        dataset = dataset.batch(32)

        return dataset

    def _binary_regression_loss(self, seg_predictions, score_factor=1.0 / 32):
        mask_target = tf.cast(self.seg_target, tf.float32)
        segmentation_loss = tf.reduce_mean(
            (1.0 + self.score_target) / 2.0 * tf.reduce_mean(tf.log(1.0 + tf.exp(-seg_predictions * mask_target)),
                                                             axis=[1, 2]))
        score_loss = tf.reduce_mean(tf.log(1.0 + tf.exp(-self.score_target * self.score_predictions))) * score_factor
        return score_loss, segmentation_loss

    def _weight_decay(self, scopes=['deepmask_trunk', 'segmentation_branch', 'score_branch']):
        weights = list(itertools.chain(*[tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope) for scope in scopes]))
        weights = list(filter(lambda x: 'kernel' in x.name, weights))
        weights_norm = tf.reduce_sum(input_tensor=tf.stack([tf.nn.l2_loss(i) for i in weights]),
                                     name='weights_norm')

        return weights_norm, weights

    def _run_validation(self, progress_ops_dict, metric_update_ops, validation_steps_count=None):
        progress_ops_names, progress_ops = zip(*progress_ops_dict.items())
        progress_ops = list(progress_ops)

        validation_ops = metric_update_ops + progress_ops

        pbar = tqdm(total=validation_steps_count, desc='Validation', file=sys.stdout)
        counter = 0

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.validation_init_op)

        while True:
            try:
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

    def _fit_cycle(self, epochs, lr_var, progress_ops_dict, opt_ops, metric_update_ops):
        progress_ops_names, progress_ops = zip(*progress_ops_dict.items())
        training_ops = opt_ops + metric_update_ops + list(progress_ops)

        train_steps_per_epoch = None
        validation_steps_per_epoch = None

        for e in range(epochs):
            tic = datetime.datetime.now()
            lr = self.sess.run([lr_var, self.training_init_op, tf.local_variables_initializer()])[0]

            print()
            tqdm.write("----- Epoch {}/{} ; learning rate {} -----".format(e + 1, epochs, lr))
            pbar = tqdm(total=train_steps_per_epoch, desc='Training', file=sys.stdout)
            train_steps_per_epoch = 0

            while True:
                try:
                    progress = self.sess.run(training_ops)[-len(progress_ops):]
                    pbar.update()
                    pbar.set_description('Training ({})'.format(
                        ', '.join(['{}={}'.format(name, val) for name, val in zip(progress_ops_names, progress)])))
                    train_steps_per_epoch += 1
                except tf.errors.OutOfRangeError as oe:
                    break

            del pbar
            validation_results = self._run_validation(progress_ops_dict, metric_update_ops, validation_steps_per_epoch)
            training_report = ', '.join(
                ['Training {}={}'.format(name, val) for name, val in zip(progress_ops_names, progress)])
            validation_report = ', '.join(
                ['Validation {}={}'.format(name, val) for name, val in validation_results.items()])
            validation_steps_per_epoch = validation_results['total_steps']
            self.saver.save(self.sess, self.checkpoint_file)
            gc.collect()
            toc = datetime.datetime.now()
            tqdm.write(
                "----- Epoch {} finished in {} -- {}. {}".format(e + 1, toc - tic, training_report, validation_report))
