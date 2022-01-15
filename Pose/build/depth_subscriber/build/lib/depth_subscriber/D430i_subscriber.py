import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from std_msgs.msg import * 
from sensor_msgs.msg import * 
import numpy as np
import message_filters

import argparse 
import json
import logging
import os
import time

import numpy as np

import cv2
import torch



from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream

from depth.mannequin.options.train_options import TrainOptions
from depth.mannequin.loaders import aligned_data_loader
from depth.mannequin.models import pix2pix_model
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
print('library loaded')

class D430i_suscriber(Node):

    def __init__(self,args):
        super().__init__('D430i_suscriber')

        # delay (in seconds) with which messages can be synchronized.
        self.slop=0.005
        #how many sets of messages it should store from each input filter (by timestamp) while waiting for messages to arrive and complete their “set”
        self.queue_size=10

        self.color_sub = message_filters.Subscriber(self, Image,"/camera/color/image_raw")
        self.depth_sub = message_filters.Subscriber(self, Image,"/camera/aligned_depth_to_color/image_raw")
        self.tss = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub],
                                                               queue_size=self.queue_size, slop=self.slop)
        self.tss.registerCallback(self.listener_callback)
        self.args=args

    def listener_callback(self, color_msg,depth_msg):
        color_plt = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
        depth_plt = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width, -1)
        fig, ax = plt.subplots()
        im = ax.imshow(depth_plt[:,:,0], cmap='gray')
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(color_plt)
        plt.show()
        print
        #Ensure that cuda cache is not used for nothing
        torch.cuda.empty_cache()
        print('cuda empty')
        Predictor.loader_workers = 1
        predictor = Predictor(
            visualize_image=(not self.args.json_output or self.args.video_output),
            visualize_processed_image=self.args.debug,
        )

        capture = Stream(self.args.source, preprocess=predictor.preprocess)
        image=capture.preprocess(color_plt)

        #We need the GT either for outputting a GT map or a doing the shifting and scaling
        if self.args.GT_depth_file is not None or self.args.shift_scale is True:
            depth_GT = depth_plt[:,:,0]
        #GT_depth_file args use the GT as the depth map for prediction
        if self.args.GT_depth_file is None or self.args.shift_scale is True :
            depth_model,transform=get_depth_model(self.args)
        print('depth model loaded')
        #We perform prediction for each frame   
        last_loop = time.perf_counter()
        (preds, _, meta)=predictor.dataset(image)
        print('preds done',preds)

        for (ax, ax_second), (preds, _, meta) in \
                zip(animation.iter(), predictor.dataset(capture)):

            image = visualizer.Base._image  # pylint: disable=protected-access
            if ax is None and (not self.args.json_output or self.args.video_output):
                ax, ax_second = animation.frame_init(image)
            
            depth_start=time.time()

            if self.args.GT_depth_file is not None and self.args.shift_scale is False :   
                depth_map=np.array(depth_GT["frame{}".format(frame_idx)])/1000
                disparity=np.array(1/depth_map)
                frame_idx=frame_idx+1
            elif self.args.GT_depth_file is None or self.args.shift_scale is True :
                depth_map, disparity, _=depth_mapping(depth_model,transform,image,self.args)
            
            #The scaling and shifting uses GT:
            if self.args.shift_scale is True :
                depth_map_GT=np.array(depth_GT["frame{}".format(frame_idx)])/1000
                disparity_GT=np.array(1/depth_map_GT)
                frame_idx=frame_idx+1
                #Threshold until when we consider good depth accuracy in meters
                depth_threshold_high=4
                depth_threshold_low=0.1
                mask=(depth_map_GT<depth_threshold_high) & (depth_map_GT>depth_threshold_low)
                mask_size=np.shape(mask)
                
                #-------------------------------------partial mask----------------------
                mask[:int(mask_size[0]/2),:]=False
                mask[:,int(mask_size[1]/2):]=False
                mask_inv=depth_map_GT>depth_threshold_high

                #We use eq(4) of Midas paper to derivate optimal scale and translation factor s and t
                square_sum=np.sum(np.power(depth_map[mask],2))
                prediction_sum=np.sum(depth_map[mask])
                shared_sum=np.sum(np.multiply(depth_map[mask],depth_map_GT[mask]))
                GT_sum=np.sum(depth_map_GT[mask])
                h_opt=np.dot(np.linalg.inv(np.array([[square_sum,prediction_sum],[prediction_sum,1]])),np.array([[shared_sum],[GT_sum]]))
                s=h_opt[0]
                t=h_opt[1]
                depth_map=depth_map*s+t
                """fig = plt.figure()
                fig.add_subplot(2, 2, 1)
                plt.imshow(depth_map_GT,cmap='gray')
                fig.add_subplot(2, 2, 2)
                depth_map_GT_plt=depth_map_GT
                depth_map_GT_plt[mask_inv]='NaN'
                plt.imshow(depth_map_GT_plt,cmap='gray')
                fig.add_subplot(2, 2, 3)
                plt.imshow(mask,cmap='gray')
                fig.add_subplot(2, 2, 4)
                plt.imshow(depth_map,cmap='gray')
                plt.show()"""
            
            depth_end=time.time()
            
            keypoints=[ann.json_data()['keypoints'] for ann in preds]
            #In midas and mannequin papers, they compute their metrics over scale invariant depth map and GT
            #Here's a reproduction of their approach on eq(6) of midas
            if self.args.normalize_pred==True:
                print('normalize_prediction')
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


            

            #if the keypoints is not found, put depth value to nan
            depth_pred[keypoint_confidence<=0.4]='nan'
            ax.imshow(disparity)
            visualizer.Base.common_ax = ax_second if self.args.separate_debug_ax else ax

            start_post = time.perf_counter()
            if self.args.json_output:
                with open(self.args.json_output, 'a+') as f:
                    json.dump({
                        'frame': meta['frame_i'],
                        'predictions': [ann.json_data() for ann in preds],
                        'depth_pred': depth_pred.tolist()
                    }, f, separators=(',', ':'))
                    f.write('\n')
            if (not self.args.json_output or self.args.video_output) \
            and (self.args.separate_debug_ax or not self.args.debug_indices):
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


def main(args=None):
    rclpy.init(args=args)
    torch.cuda.empty_cache()
    args = cli()
    minimal_subscriber = D430i_suscriber(args)
    rclpy.spin(minimal_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()





#--------------------------------------------------------------------------------






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
    parser.add_argument('--video-output', default='output/ROS2.mp4', nargs='?', const=True,
                        help='video output file or "virtualcam"')
    parser.add_argument('--json-output', default='output/ROS2.json', nargs='?', const=True,
                        help='json output file')
    parser.add_argument('--depth_model', default='midas', nargs='?', const=True,
                        help='Network used for depth estimation : [mannequin|midas]')
    parser.add_argument('--separate-debug-ax', default=False, action='store_true')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--normalize-pred', action='store_true',
                        help='normalize prediction to reproduce papers')
    parser.add_argument('--GT_depth_file', type=str, default=None,
                        help='depth map ground truth')
    parser.add_argument('--shift-scale', default=True,
                        help='Use ground truth to fit depth scaling and shift')
    
    args = parser.parse_args()

    midas_depth_path='/data/drame/openpifpaf1.3/openpifpaf/depth/midas/weights'
    #If no model_weights are defined, we'll take this one as default 
    default_models = {
        "midas_v21_small": midas_depth_path +"/midas_v21_small-70d6b9c8.pt",
        "midas_v21": midas_depth_path +"/midas_v21-f6b98070.pt",
        "dpt_large": midas_depth_path +"/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": midas_depth_path +"/dpt_hybrid-midas-501f0c75.pt",
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

        
    else:
        print('Wrong depth model name, try : [mannequin|midas]')

    return pred_d, disparity, end-start