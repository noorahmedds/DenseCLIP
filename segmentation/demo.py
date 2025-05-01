# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


import denseclip

import numpy as np
np.random.seed(11)

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--checkpoint', default=None, help='The demo requires that a checkpoint is provided. Use this similar to the pretrained parameter inside yor cfg')
    parser.add_argument(
        '--class_names',
        nargs='+',
        default=['Bicycle Body', 'Bicycle Head', 'Bicycle Seat', 'Bicycle Tire'],
        help='List of class names to be used for segmentation')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=args.checkpoint, device=args.device)

    if args.class_names is not None:
        model._init_classes(args.class_names)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    # test a single image
    result = inference_segmentor(model, args.img)

    palette = np.random.randint(0,255,(len(model.CLASSES), 3))
    
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        palette,
        opacity=args.opacity,
        title="Hand Demo")


if __name__ == '__main__':
    main()