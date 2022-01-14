"""Video demo application.

Use --scale=0.2 to reduce the input image size to 20%.
Use --json-output for headless processing.

Example commands:
    python3 -m pifpaf.video --source=0  # default webcam
    python3 -m pifpaf.video --source=1  # another webcam

    # streaming source
    python3 -m pifpaf.video --source=http://127.0.0.1:8080/video

    # file system source (any valid OpenCV source)
    python3 -m pifpaf.video --source=docs/coco/000000081988.jpg

Trouble shooting:
* MacOSX: try to prefix the command with "MPLBACKEND=MACOSX".
"""


import argparse 
import json
import logging
import os
import time

import torch

from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream

from depth.mannequin.options.train_options import TrainOptions
from depth.mannequin.loaders import aligned_data_loader
from depth.mannequin.models import pix2pix_model
import cv2
import numpy as np
import torch.autograd as autograd
from skimage import transform
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from torchvision.transforms import Compose

from depth.midas import midas_cli
from depth.midas.midas.dpt_depth import DPTDepthModel
from depth.midas.midas.midas_net import MidasNet
from depth.midas.midas.midas_net_custom import MidasNet_small
from depth.midas.midas.transforms import Resize, NormalizeImage, PrepareForNet

LOG = logging.getLogger(__name__)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.video',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.Factory.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    Predictor.cli(parser)
    show.cli(parser)
    Stream.cli(parser)
    visualizer.cli(parser)
    midas_cli.cli(parser)

    parser.add_argument('--source', default='0',
                        help=('OpenCV source url. Integer for webcams. '
                              'Or ipwebcam urls (rtsp/rtmp). '
                              'Use "screen" for screen grabs.'))
    parser.add_argument('--video-output', default=None, nargs='?', const=True,
                        help='video output file or "virtualcam"')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='json output file')
    parser.add_argument('--depth_model', default='mannequin', nargs='?', const=True,
                        help='Network used for depth estimation : [mannequin|midas]')
    parser.add_argument('--separate-debug-ax', default=False, action='store_true')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--normalize-pred', action='store_true',
                        help='normalize prediction to reproduce papers')
    
    args = parser.parse_args()

    #If no model_weights are defined, we'll take this one as default 
    default_models = {
        "midas_v21_small": "depth/midas/weights/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "depth/midas/weights/midas_v21-f6b98070.pt",
        "dpt_large": "depth/midas/weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "depth/midas/weights/dpt_hybrid-midas-501f0c75.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    Stream.configure(args)
    visualizer.configure(args)


    # check whether source should be an int
    if len(args.source) == 1:
        args.source = int(args.source)

    # standard filenames
    if args.video_output is True:
        args.video_output = '{}.openpifpaf.mp4'.format(args.source)
        if os.path.exists(args.video_output):
            os.remove(args.video_output)
    assert args.video_output is None or not os.path.exists(args.video_output)
    if args.json_output is True:
        args.json_output = '{}.openpifpaf.json'.format(args.source)
        if os.path.exists(args.json_output):
            os.remove(args.json_output)
    assert args.json_output is None or not os.path.exists(args.json_output)
    assert os.path.exists(args.source), "Source file doesn't exist"

    return args

def get_depth_model(args):
    if args.depth_model=='mannequin':
        opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

        eval_num_threads = 2
        model = pix2pix_model.Pix2PixModel(opt)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        #Transformation in mannequin are implemented in the depth mapping
        transform=False
    elif args.depth_model=='midas':
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

            # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: %s" % device)

        # load network
        if args.model_type == "dpt_large": # DPT-Large
            model = DPTDepthModel(
                path=args.model_weights,
                backbone="vitl16_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif args.model_type == "dpt_hybrid": #DPT-Hybrid
            model = DPTDepthModel(
                path=args.model_weights,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode="minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif args.model_type == "midas_v21":
            model = MidasNet(args.model_weights, non_negative=True)
            net_w, net_h = 384, 384
            resize_mode="upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif args.model_type == "midas_v21_small":
            model = MidasNet_small(args.model_weights, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
            resize_mode="upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            print(f"args.model_type '{args.model_type}' not implemented, use: --args.model_type large")
            assert False
        
        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()
        
        if args.optimize==True:
            # rand_example = torch.rand(1, 3, net_h, net_w)
            # model(rand_example)
            # traced_script_module = torch.jit.trace(model, rand_example)
            # model = traced_script_module
        
            if device == torch.device("cuda"):
                model = model.to(memory_format=torch.channels_last)  
                model = model.half()

        model.to(device)
    else:
        print('Wrong depth model name, try : [mannequin|midas]')
    return model,transform

def depth_mapping(model,transformation,frame,args):
    input_height,input_width,_=np.shape(frame)   

    if args.depth_model=='mannequin':

        height_depthmap = 288
        width_depthmap = 512

        #resize for the model
        frame=torch.tensor(frame)

        frame = np.float32(frame)/255.0
        frame = transform.resize(frame, (width_depthmap,height_depthmap))
        frame=torch.tensor(frame)

        frame=torch.transpose(frame,0,2)
        frame=torch.transpose(frame,1,2).unsqueeze(0)

        #run the model
        start = time.time()
        prediction_d, pred_confidence = model.netG.forward(frame)
        pred_log_d = prediction_d.squeeze(1)
        pred_d = torch.exp(pred_log_d)

        end = time.time()

        pred_d_ref = pred_d.data[0, :, :].cpu().numpy()
        disparity = 1. / pred_d_ref

        disparity = disparity / np.max(disparity)
        disparity = np.tile(np.expand_dims(disparity, axis=-1), (1, 1, 3))

        disparity = (disparity*255).astype(np.uint8)

        pred_d=torch.transpose(pred_d,0,2)
        pred_d=torch.transpose(pred_d,1,0)
        pred_d=pred_d.detach().numpy()
        pred_d = cv2.resize(pred_d, (input_width,input_height))
        disparity = cv2.resize(disparity, (input_width,input_height))
        # fig, ax = plt.subplots()
        # im = ax.imshow(disparity, cmap='gray', vmin=0, vmax=255)
        # plt.show()

    elif args.depth_model=='midas':
        # input
        start=time.time()
        if frame.ndim == 2:
            print('frame.ndim == 2')
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        

        frame = frame / 255.0
        # fig, ax = plt.subplots()
        # im = ax.imshow(frame)
        # plt.show()

        img_input = transformation({"image": frame})["image"]
        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(args.device).unsqueeze(0)
            if args.optimize==True and args.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)  
                sample = sample.half()
            disparity = model.forward(sample)
            disparity = (
                torch.nn.functional.interpolate(
                    disparity.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy().astype(np.float32)
            )
        end=time.time()
        depth_max=np.max(disparity)
        depth_min=np.min(disparity)

        disparity[disparity<0.001]=0.001
        pred_d = 1. / disparity

        disparity = disparity / np.max(disparity)
        #print(disparity)
        disparity = np.tile(np.expand_dims(disparity, axis=-1), (1, 1, 3))

        disparity = (disparity*255).astype(np.uint8)

        pred_d = cv2.resize(pred_d, (input_width,input_height))
        disparity = cv2.resize(disparity, (input_width,input_height))
        # fig, ax = plt.subplots()
        # im = ax.imshow(pred_d, cmap='gray', vmin=0, vmax=255)
        # plt.show()
        
    else:
        print('Wrong depth model name, try : [mannequin|midas]')



    return pred_d, disparity, end-start

def main():
    args = cli()

    Predictor.loader_workers = 1
    predictor = Predictor(
        visualize_image=(not args.json_output or args.video_output),
        visualize_processed_image=args.debug,
    )
    capture = Stream(args.source, preprocess=predictor.preprocess)

    annotation_painter = show.AnnotationPainter() 
    animation = show.AnimationFrame(
        video_output=args.video_output,
        second_visual=args.separate_debug_ax,
    )

    depth_model,transform=get_depth_model(args)
    last_loop = time.perf_counter()
    for (ax, ax_second), (preds, _, meta) in \
            zip(animation.iter(), predictor.dataset(capture)):

        image = visualizer.Base._image  # pylint: disable=protected-access
        if ax is None and (not args.json_output or args.video_output):
            ax, ax_second = animation.frame_init(image)
        depth_start=time.time()
        depth_map, disparity, _=depth_mapping(depth_model,transform,image,args)
        depth_end=time.time()
        
        keypoints=[ann.json_data()['keypoints'] for ann in preds]
        #In midas and mannequin papers, they compute their metrics over scale invariant depth map and GT
        #Here's a reproduction of their approach
        if args.normalize_pred==True:
            t=np.median(depth_map)
            s=np.mean(np.abs(depth_map-t))
            depth_map=(depth_map-t)/s

        
        if len(preds)>0:
            preds_indice=np.reshape(keypoints,(len(preds),17,-1)).astype(int)
            keypoint_confidence=np.reshape(keypoints,(len(preds),17,-1))[:,:,2]
            preds_indice[:,:,[0,1]] = np.abs(preds_indice[:,:,[1,0]])
            nb_predictions=0

        else:
            preds_indice=[]
            keypoint_confidence=np.zeros((len(preds),17))

        depth_pred=np.zeros((len(preds),17))
        for idx,keypoints_single in enumerate(preds_indice) :

            one=np.ones(np.shape(keypoints_single[:,[0,1]].transpose().tolist())).astype(int)

            depth_pred[nb_predictions]=depth_map[tuple(keypoints_single[:,[0,1]].transpose().tolist()-one)]
            nb_predictions+=1
            depth_map[tuple(keypoints_single[:,[0,1]].transpose().tolist()-one)]=1


        # plt.imshow(depth_map,cmap='gray')
        # plt.show()

        #if the keypoints is not found, put depth value to nan
        depth_pred[keypoint_confidence<=0.4]='nan'
        ax.imshow(disparity)
        visualizer.Base.common_ax = ax_second if args.separate_debug_ax else ax

        start_post = time.perf_counter()
        if args.json_output:
            with open(args.json_output, 'a+') as f:
                json.dump({
                    'frame': meta['frame_i'],
                    'predictions': [ann.json_data() for ann in preds],
                    'depth_pred': depth_pred.tolist()
                }, f, separators=(',', ':'))
                f.write('\n')
        if (not args.json_output or args.video_output) \
           and (args.separate_debug_ax or not args.debug_indices):
            depth_text=np.nanmean(depth_pred,axis=1)
            annotation_painter.annotations(ax, preds,texts=depth_text)

        postprocessing_time = time.perf_counter() - start_post
        if animation.last_draw_time is not None:
            postprocessing_time += animation.last_draw_time

        LOG.info('frame %d, total loop time = %.0fms total FPS = %.4f, \n open loop time %.0f ms,open fps  %.4f Hz, \n depth loop  %.0f ms, depth fps  %.4f Hz\n',
                meta['frame_i'],
                (time.perf_counter() - last_loop) * 1000.0,
                1.0 / (time.perf_counter() - last_loop),
                (time.perf_counter() - last_loop) * 1000.0-(depth_end-depth_start)*1000.0,
                1.0/((time.perf_counter() - last_loop) -(depth_end-depth_start)),
                (depth_end-depth_start)*1000.0,
                1.0/( depth_end-depth_start))
                 
        last_loop = time.perf_counter()

#meta['preprocessing_s'] * 1000.0,
#postprocessing_time * 1000.0,
if __name__ == '__main__':
    main()
