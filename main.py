import sys
import argparse
from sharpmask import SharpMask


def main(eval=False):
    parser = argparse.ArgumentParser(
        description='Use this util to prepare tfrecord files before training DeepMask/SharpMask')

    if eval:
        # dm = SharpMask(train_path='D:/data/coco/tfrecord', validation_path='D:/data/coco/tfrecord_val',
        #                resnet_ckpt="D:\\data\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary", checkpoint_path='D:/data/coco/sm_model')
        dm = SharpMask(train_path='D:/data/coco/tfrecord', validation_path='D:/data/coco/tfrecord_val',
                       resnet_ckpt="D:\\data\\coco\\resnet_chk\\model.ckpt-250200", summary_path="./summary",
                       checkpoint_path='D:/data/coco/sm_model')
        # dm.restore()
        # dm.eval('E:\\data\\ml\\coco\\images\\val2017\\000000001584.jpg')
        # dm.eval_resnet('bear.jpg')
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
        # dm.eval_resnet('test_images/bear.jpg')

        dm.fit_deepmask()


if __name__ == "__main__":
    sys.exit(main())