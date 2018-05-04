import sys
import argparse
from sharpmask import SharpMask


def main():
    parser = argparse.ArgumentParser(
        description='DeepMask/SharpMask TensorFlow implementation')

    parser.add_argument('--model', action="store", dest="model", default='all',
                        choices=['all', 'deepmask', 'sharpmask'],
                        help='The model: deepmask, sharpmask or both',
                        required=False)

    parser.add_argument('--restore', action="store_true", dest="restore",
                        help='Restore model from checkpoint', required=False)

    parser.add_argument('--train', action="store_true", dest="train",
                        help='Run the model training. For sharpmask model only refinement layers training is performed.',
                        required=False)

    parser.add_argument('--validate', action="store_true", dest="validate",
                        help='Run explicit validation of selected model on validation set provided in the validation_path argument',
                        required=False)

    parser.add_argument('--evaluate', action="store_true", dest="evaluate",
                        help='Evaluate the model applying mask on an image provided by the eval_source argument',
                        required=False)

    parser.add_argument('--train_path', action="store", dest="train_path",
                        help='A path to folder containing train set tfrecord files', required=False)

    parser.add_argument('--validation_path', action="store", dest="validation_path",
                        help='A path to folder containing validation set tfrecord files', required=False)

    parser.add_argument('--resnet_ckpt', action="store", dest="resnet_ckpt",
                        help='A path to pretrained resnet-50 checkpoint', required=False)

    parser.add_argument('--summary_path', action="store", dest="summary_path", help='A path to store model summary',
                        required=False)

    parser.add_argument('--checkpoint_path', action="store", dest="checkpoint_path",
                        help='A path to store model checkpoint',
                        required=False)

    parser.add_argument('--eval_source', action="store", dest="eval_source", help='Source image for evalutation',
                        required=False)

    parser.add_argument('--eval_target', action="store", dest="eval_target",
                        help='Target file name for image with applied mask', required=False)

    params = parser.parse_args(sys.argv[1:])
    print(params)

    if not (params.train or params.validate or params.evaluate):
        print('To run the model at least one option from train, validate or evaluate should be chosen.',
              file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(-1)

    model = SharpMask(train_path=params.train_path, validation_path=params.validation_path,
                      resnet_ckpt=params.resnet_ckpt, summary_path=params.summary_path,
                      checkpoint_path=params.checkpoint_path)

    if params.restore:
        model.restore()

    if params.train:
        if params.model == 'deepmask' or params.train == 'all':
            model.fit_deepmask()
        if params.model == 'sharpmask' or params.model == 'all':
            model.fit_sharpmask()

    if params.validate:
        if params.model == 'deepmask':
            model.deepmask_validation()
        if params.model == 'sharpmask':
            model.sharpmask_validation()

    if params.evaluate:
        if params.model == 'deepmask':
            model.eval(params.eval_source)
        if params.model == 'sharpmask':
            model.eval(params.eval_source)

    # if eval:
    #     # dm = SharpMask(train_path='D:/data/coco/tfrecord', validation_path='D:/data/coco/tfrecord_val',
    #     #                resnet_ckpt="D:\\data\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary", checkpoint_path='D:/data/coco/sm_model')
    #     dm = SharpMask(train_path='D:/data/coco/tfrecord', validation_path='D:/data/coco/tfrecord_val',
    #                    resnet_ckpt="D:\\data\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary",
    #                    checkpoint_path='D:/data/coco/sm_model')
    #     # dm.restore()
    #     # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000001584.jpg')
    #     # dm.eval_resnet('bear.jpg')
    #     # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000000785.jpg')
    #     # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000000285.jpg')
    #     # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000007784.jpg')
    #
    #     # dm.eval('nig.jpg')
    # else:
    #     dm = SharpMask(train_path='E:/data/ml/coco/tfrecord_224', validation_path='E:/data/ml/coco/tfrecord_val_224',
    #                    resnet_ckpt="E:\\data\\ml\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary",
    #                    checkpoint_path="E:/data/ml/coco/sm_dump")
    #     # dm = SharpMask(train_path='D:/data/coco/tfrecord_val_224', validation_path='D:/data/coco/tfrecord_val_224',
    #     #               resnet_ckpt="D:\\data\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary", checkpoint_path="D:/data/coco/sm_dump")
    #     # dm.restore()
    #     # dm.eval_resnet('test_images/bear.jpg')
    #
    #     #dm.fit_deepmask()


if __name__ == "__main__":
    sys.exit(main())
