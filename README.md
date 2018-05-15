SharpMask
=========
DeepMask[1] and SharpMask[2] implementation on Python/TensorFlow.

Introduction
------------
The repository contains an implementation of DeepMask and SharpMask models. 
DeepMask model predicts class agnostic object mask and object score which is positive if an object is centered and fully contained in an image. SharpMask is an extension of DeepMask architecture, which uses 
a top-down refinement module to compute more precise object mask proposal.

The implementation is based on TensorFlow official [ResNet-v2](https://github.com/tensorflow/models/tree/master/official/resnet)[3] model implementation and
requires pre-trained ResNet [weights](http://download.tensorflow.org/models/official/resnet_v2_imagenet_checkpoint.tar.gz "TensorFlow ResNet-v2 checkpoint").

ResNet model implementation is copied from the official TensorFlow repository.

Training
----------

Evaluation
----------

Examples
----------

Pre-trained weights
------------------


References
----------
[1]: https://arxiv.org/abs/1603.08695 "Pedro O. Pinheiro et al.: Learning to Segment Object Candidates"
[2]: https://arxiv.org/abs/1506.06204 "Pedro O. Pinheiro et al.: Learning to Refine Object Segments"
[3]: https://arxiv.org/abs/1512.03385 "Kaiming He et al.: Deep Residual Learning for Image Recognition"