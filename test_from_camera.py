from numpy.core.numeric import NaN
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from std_msgs.msg import * 
from sensor_msgs.msg import * 
import numpy as np
import message_filters
#create cloud libraries 
from collections import namedtuple
import ctypes
#import math
import struct

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

import open3d as o3d
import pyrealsense2 as rs

nose        =  0
left_eye    =  1
right_eye   =  2
left_ear    =  3
right_ear   =  4
left_shoulder = 5 
right_shoulder = 6 
left_elbow  =  7
right_elbow =  8
left_wrist  =  9
right_wrist =  10
left_hip    =  11
right_hip   =  12
left_knee   =  13
right_knee  =  14
left_ankle  =  15
right_ankle  =  16

coco_kp=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
         'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee',
         'left_ankle','right_ankle']

pairs = [[nose, left_eye], [nose, right_eye], [left_eye, right_eye], 
         [left_eye, left_ear],[left_ear, left_shoulder],[right_eye, right_ear], [right_ear, right_shoulder],
         [left_shoulder, right_shoulder], [left_shoulder, left_hip], [right_shoulder, right_hip], [left_hip,right_hip],
         [left_shoulder, left_elbow], [left_elbow, left_wrist], [right_shoulder, right_elbow], [right_elbow, right_wrist],
         [left_hip,left_knee], [left_knee,left_ankle], [right_hip, right_knee], [right_knee, right_ankle] 
]
colors = ['r', 'g', 'b', 'y', 'k', 'p']
colors_rgb=[(255,0,0),(0,0,255),(0,0,255),(255,255,0),(0,0,0),(255,105,180)]

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile=self.pipeline.start(config)


        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale) 
        
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()


def main(args=None):
    rclpy.init(args=args)
    torch.cuda.empty_cache()
    args = cli()
    args.checkpoint='mobilenetv2'
    args.shift_scale=True
    args.model_weigth='dpt_hybrid' # Midas model dpt_hybrid, midas_v21_large or midas_v21_small'
    # Set true for real time plot and false for single plot a time
    args.plot_pointcloud=False
    args.project_pointcloud_torso_frame=False
    args.plot_real_time=True

    args.plot_skeleton=True
    # Set true to use camera info as ground truth
    args.GT_from_camera=True
    #Ensure that cuda cache is not used for nothing
    torch.cuda.empty_cache()
    Predictor.loader_workers = 1
    predictor = Predictor(
        visualize_image=(not args.json_output or args.video_output),
        visualize_processed_image=args.debug,
    )

    annotation_painter = show.AnnotationPainter() 
    animation = show.AnimationFrame(
        video_output=args.video_output,
        second_visual=args.separate_debug_ax,
    )
    capture = Stream(args.source, preprocess=predictor.preprocess)
    depth_model,transform=get_depth_model(args)

    #Set the visualizer for depth cloud
    if args.plot_pointcloud:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
    else :
        vis=None

    frame_index=0

    dc = DepthCamera() 
    while(True): 
        start_callback = time.perf_counter()
        torch.cuda.empty_cache()
        ret, depth_plt, color_plt = dc.get_frame()

        color_plt[:,:,[0,1,2]]=color_plt[:,:,[2,1,0]]
        # fig, ax = plt.subplots()
        # im = ax.imshow(depth_plt, cmap='gray')
        # fig2, ax2 = plt.subplots()
        # im2 = ax2.imshow(color_plt)
        # plt.show()
        start_open = time.perf_counter()
        
        image, processed_image, anns, meta = capture.preprocessing(torch.tensor(color_plt))

        #We resize the input in openpifpaf networks input dimension.
        cap=[[image,torch.tensor(np.expand_dims(processed_image.numpy(),0)), [[]], np.expand_dims(meta,0)]]#(np.expand_dims(anns,0)

        preds=predictor.dataloader(cap)
        #pred is a generator containing pred, gt_anns, meta, we acces the annotation as follow
        pred=(next(preds)[0])

        keypoints=[ann.json_data()['keypoints'] for ann in pred] #json_data()
        #print('pred',pred,len(pred))
        if len(pred)>0:
            preds_indice=np.reshape(keypoints,(len(pred),17,-1)).astype(int)
            keypoint_confidence=np.reshape(keypoints,(len(pred),17,-1))[:,:,2]
            preds_indice = np.abs(preds_indice[:,:,0:2])
            preds_indice[:,:,[0,1]] = np.abs(preds_indice[:,:,[1,0]])
            nb_predictions=0
        else:
            preds_indice=[]
            keypoint_confidence=np.zeros((len(pred),17))
        end_open = time.perf_counter()
        start_depthmap = time.perf_counter()
        if args.GT_from_camera is False:
            depth_map, disparity, _=depth_mapping(depth_model,transform,color_plt,args)
            fig3, ax3 = plt.subplots()
            im3 = ax3.imshow(depth_map,cmap='gray')
        elif args.GT_from_camera :
            depth_map=depth_plt/1000
        
        #The scaling and shifting uses GT:
        if args.shift_scale and args.GT_from_camera is False :
            #We need the GT either for outputting a GT map or a doing the shifting and scaling
            depth_GT = depth_plt
            depth_map_GT=np.array(depth_GT)/1000 #in meters
            disparity_GT=np.array(1/depth_map_GT)
            #Threshold until when we consider good depth accuracy in meters
            depth_threshold_high=4.5
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
            #print('scaling and translation coef',s,t)
            depth_map=depth_map*s+t
            #depth_map[depth_map>10]=0        

        end_depthmap = time.perf_counter()

        width,heigth,_=np.shape(color_plt)

        if len(preds_indice)>0:
            color_idx=0
            for people_kp in preds_indice :
                for kp in people_kp:
                    if any(np.array(kp)==0) :
                        continue
                    else:
                        color_plt=cv2.circle(color_plt,tuple([kp[1],kp[0]]), 5, (255,105,180),-1) 
                for pair in pairs :   
                    if (np.array(people_kp[pair])==0).any() :
                        continue
                    color_plt=cv2.line(color_plt,people_kp[pair[1]][[1,0]], people_kp[pair[0]][[1,0]], (255,0,0),3)     
                color_idx+=1

              
        
        #These are the intrisic of the camera (in pxl), they can be obtain on the P array of the camera info topics
        #http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        fx=909.2
        fy=909.0
        cx=653.7
        cy=367.6
        s=0
        #color_plt=cv2.circle(color_plt,tuple([int(cx),int(cy)]), 10, (0,255,0),-1) 
        

        #For each person we predict the depth
        depth_pred=np.zeros((len(preds_indice),17))
        for people_idx,people_keypoints in enumerate(preds_indice) :
            for bodypart_idx,bodypart_keypoints in enumerate(people_keypoints) :
                depth_pred[people_idx,bodypart_idx]=depth_map[tuple(bodypart_keypoints-1)]
        depth_pred[keypoint_confidence<=0.4]=0

        #inverse of P, the Projection/camera matrix
        #https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
        P_inv=np.array([[1/fx, -s/(fx*fy), (s*cy-cx*fx)/(fx*fy)],
                        [0, 1/fy, -cy/fy],
                        [0, 0, 1] ] )
        #We now convert our pixel value to value in meter for x y position               
        cartesian_pred=np.zeros((len(preds_indice),3,17))
        for idx,(depth_single,keypoints_single) in enumerate(zip(depth_pred,preds_indice)):
            cartesian_pred[idx]=np.array([depth_single*np.array(P_inv@[keypoints_single[:,1],keypoints_single[:,0],np.ones_like(keypoints_single[:,0])]) ])

        depth_pred[keypoint_confidence<=0.4]=np.nan
        cartesian_pred[cartesian_pred==0.]=np.nan

        if len(pred)>0:
            # print('shoulder shoulder dist',np.sqrt((cartesian_pred[0,0,left_shoulder]-cartesian_pred[0,0,right_shoulder])**2+(cartesian_pred[0,1,left_shoulder]-cartesian_pred[0,1,right_shoulder])**2))
            # print('hips hips dist',np.sqrt((cartesian_pred[0,0,left_hip]-cartesian_pred[0,0,right_hip])**2+(cartesian_pred[0,1,left_hip]-cartesian_pred[0,1,right_hip])**2))
            # print('left shoulder hips dist',np.sqrt((cartesian_pred[0,0,left_shoulder]-cartesian_pred[0,0,left_hip])**2+(cartesian_pred[0,1,left_shoulder]-cartesian_pred[0,1,left_hip])**2))
            # print('right shoulder hips dist',np.sqrt((cartesian_pred[0,0,right_shoulder]-cartesian_pred[0,0,right_hip])**2+(cartesian_pred[0,1,right_shoulder]-cartesian_pred[0,1,right_hip])**2))

            #We want to compute body orientation in the body reference frame
            Torso_coordinate=np.array([cartesian_pred[0,:,left_shoulder],cartesian_pred[0,:,right_shoulder],cartesian_pred[0,:,left_hip],cartesian_pred[0,:,right_hip]])
            Torso_coordinate=Torso_coordinate[~np.isnan(Torso_coordinate)]
            Torso_coordinate=np.reshape(Torso_coordinate,(int(len(Torso_coordinate)/3),3))
        else :
            Torso_coordinate=[]

        start_rotation= time.perf_counter()

        #We need at least three keypoint of the torso to do a projection on the torso frame
        if len(Torso_coordinate)<4:
            print('At least three point need to be located on the torso to project to the torso frame - Camera frame is used.')
            R=np.eye(3)
            T=np.array([0,0,0])
            Torso_centroid=np.array([0,0,0])
        else :
            #Center of the new frame
            Torso_centroid=np.mean(Torso_coordinate,0)

            #3d approach doesn't work well yet
            """Torso_centered_xyz=Torso_coordinate-Torso_centroid

            #We make sure that the left shoulder is always on the positive axis and that y axis goes in the direction from shoulder to hips
            x, y, z =0,1,2
            shoulder, hips=[0,1], [2,3]
            left, right=[0,2], [1,3]

            print(len(Torso_centered_xyz))
            Torso_variance_xyz=np.dot((Torso_centered_xyz).transpose(),(Torso_centered_xyz))/len(Torso_centered_xyz)
            eigen_value_xyz,eigenvector_xyz=np.linalg.eig(Torso_variance_xyz)

            new_coordinate_xyz=np.dot(eigenvector_xyz,Torso_centered_xyz.transpose())

            R,T=rigid_transform_3D(np.array(Torso_coordinate.transpose()),new_coordinate_xyz)"""

            #print('Rotation : \n',R,'\nTranslation : \n',T,'\neigenvector\n',eigenvector_xyz,'\neigenvalue\n',eigen_value_xyz)
            #We  want to have the x vector on the shoulder (bigger eigenvalue, data are more spread along the shoulder axis )   

            #We shearch the principal axis for the xz plan, we want to keep y pointb up in camera frame 
            Torso_centered_xz=Torso_coordinate[:,[0,2]]-Torso_centroid[[0,2]]
            Torso_centered_y=Torso_coordinate[:,1]-Torso_centroid[1]
            #We make sure that the left shoulder is always on the positive axis and that y axis goes in the direction from shoulder to hips
            x,z,=0,1
            shoulder,hips=[0,1],[2,3]
            left,right=[0,2],[1,3]
            if np.mean(Torso_centered_xz[left,x])<np.mean(Torso_centered_xz [right,x]):
                Torso_centered_xz[left,x]=-Torso_centered_xz[left,x]
                Torso_centered_xz[right,x]=-Torso_centered_xz[right,x]
                print('left right alternated')
            if np.mean(Torso_centered_xz[shoulder,z,])>np.mean(Torso_centered_xz[hips,z,]):
                Torso_centered_xz[shoulder,z,]=-Torso_centered_xz[shoulder,z,]
                Torso_centered_xz[hips,z,]=-Torso_centered_xz[hips,z,]
                print('shoulder hips alternated')
            #print('torso coord',Torso_coordinate[:,[0,2]],'torso centroid',Torso_centroid[[0,2]])
            Torso_variance_xz=np.dot((Torso_centered_xz).transpose(),(Torso_centered_xz))/len(Torso_centered_xz)

            #The eigenvector of the covariance matrix, gives us the pricipals axis where we will maximize information. Meaning the axis align with the body 
            eigen_value,eigenvector=np.linalg.eig(Torso_variance_xz)
            #We sort from bigger to smaller value
            sorting=np.argsort(-1*eigen_value)
            #We  want to have the x vector on the shoulder (bigger eigenvalue, data are more spread along the shoulder axis )
            eigenvector=eigenvector.transpose()[sorting].transpose()
            #On X,Z it's the projection of the point on the new referentiel, on the Y axis we just invert the axis.
            new_coordinate=np.dot(eigenvector,Torso_centered_xz.transpose())

            R,T=rigid_transform_3D(np.array(Torso_coordinate.transpose()[[0,2]]),new_coordinate)

            R_tmp=np.zeros((3,3))
            R_tmp[tuple([[0,0,2,2],[0,2,0,2]])]=np.reshape(eigenvector,(1,4))
            R_tmp[1,1]=-1
            R=R_tmp
            T=np.array([[T.item(0)],[Torso_centroid[1]],[T.item(1)]])


            # print('new_coordinate\n',new_coordinate)
            # print('\n\nleft shoulder x,y :\n', new_coordinate[:,0])
            # print('right shoulder x,y :\n', new_coordinate[:,1])
            # print('left hip x,y :\n', new_coordinate[:,2])
            # print('right hip x,y :\n', new_coordinate[:,3],'\n\n')
            # print('Torso_centered_xz\n',Torso_centered_xz)

            # print('cartesian_pred\n',cartesian_pred,'\n--------------------------')
            # print('Torso_variance_xz\n\n',Torso_variance_xz,'\nlen torso coord\n\n',len(Torso_coordinate))
            # print('eigen_value\n\n',eigen_value,'\neigenvector\n\n',eigenvector)
            # print('new coord xzy - y pos\n\n', np.concatenate((np.dot(eigenvector,Torso_centered_xz.transpose()),np.array([(Torso_coordinate[:,1]-Torso_centroid[1])]) ),axis=0))
            # print('new coord xzy\n', np.concatenate((np.dot(eigenvector,Torso_centered_xz.transpose()),np.array([-(Torso_coordinate[:,1]-Torso_centroid[1])]) ),axis=0))
            #print('new_coordinate\n',new_coordinate)
            # print(np.shape(Torso_coordinate),np.shape(new_coordinate))           
            #print('Rotation : \n',R,'\n Translation : \n',T)
            
            # print('Projection\n',R@Torso_coordinate.transpose()+T)
            # print('Projection referentiel\n',R@np.eye(3).transpose())

            # print('Error :' ,(R@Torso_coordinate.transpose()+T)-(new_coordinate))
            # print('cart_pred reproj',R@cartesian_pred+T)
            
            
            # if np.mean(torso_projection[x,shoulder])<np.mean(torso_projection[x,hips]):
            #     R[x,:]*=-1
            #     T[x]*=-1
            #     print('shoulder hips on x axis alternated')
            # if np.mean(torso_projection[y,left])<np.mean(torso_projection[y,right]):
            #     R[y,:]*=-1
            #     T[y]*=-1
            #     print('left right on y axis alternated')
            # print('R\n',R,'\neigeinvalue\n',eigen_value_xyz,'\neigenvector_xyz\n',eigenvector_xyz)
            # torso_projection=R@(Torso_coordinate.transpose())+T
            # print('torso_projection\n',torso_projection)


            cartesian_pred=R.transpose()@(cartesian_pred)+T
     
            
            #Plot of torso frame and camera frame
            # fig_torso_frame = plt.figure(figsize=(20.0, 10.0))
            # ax1 = fig_torso_frame.add_subplot(121,projection='3d')

            # ax1.plot(np.concatenate((new_coordinate_xyz[x,[0,1,3,2]],new_coordinate_xyz[x,[0]])), np.concatenate((new_coordinate_xyz[y,[0,1,3,2]],new_coordinate_xyz[y,[0]])), np.concatenate((new_coordinate_xyz[z,[0,1,3,2]],new_coordinate_xyz[z,[0]])), color='g')
            # ax1.plot(np.concatenate((Torso_centered_xyz[[0,1,3,2],x],Torso_centered_xyz[[0],x])), np.concatenate((Torso_centered_xyz[[0,1,3,2],y],Torso_centered_xyz[[0],y])),np.concatenate((Torso_centered_xyz[[0,1,3,2],z],Torso_centered_xyz[[0],z])), color='y')

            # ax1.scatter(new_coordinate_xyz[x,0], new_coordinate_xyz[y,0], new_coordinate_xyz[z,0], marker='o',color='r')
            # ax1.scatter(new_coordinate_xyz[x,1], new_coordinate_xyz[y,1], new_coordinate_xyz[z,1], marker='o',color='m')
            # ax1.scatter(new_coordinate_xyz[x,2], new_coordinate_xyz[y,2], new_coordinate_xyz[z,2], marker='o',color='b')
            # ax1.scatter(new_coordinate_xyz[x,3], new_coordinate_xyz[y,3], new_coordinate_xyz[z,3], marker='o',color='c')
            # ax1.plot([0,0],[0,0.1],color='r')
            # ax1.plot([0,0.1],[0,0],color='r')
            # ax1.plot([0,0],[0,0],[0,0.1],color='r')

            # ax1.scatter(Torso_centered_xyz[0,x], Torso_centered_xyz[0,y],Torso_centered_xyz[0,z], marker='x',color='r')
            # ax1.scatter(Torso_centered_xyz[1,x], Torso_centered_xyz[1,y],Torso_centered_xyz[1,z], marker='x',color='m')
            # ax1.scatter(Torso_centered_xyz[2,x], Torso_centered_xyz[2,y],Torso_centered_xyz[2,z], marker='x',color='b')
            # ax1.scatter(Torso_centered_xyz[3,x], Torso_centered_xyz[3,y],Torso_centered_xyz[3,z], marker='x',color='c')
            # ax1.set_xlabel('X [m]')
            # ax1.set_ylabel('Z [m]')
            # ax1.set_zlabel('Y [m]')
            # ax1.view_init(azim=145, elev=-140)

            # ax2 = fig_torso_frame.add_subplot(122,projection='3d')

            # ax2.plot(np.concatenate((new_coordinate[x,[0,1,3,2]],new_coordinate[x,[0]])), np.concatenate((new_coordinate[y,[0,1,3,2]],new_coordinate[y,[0]])), np.zeros((5)), color='g')
            # ax2.plot(np.concatenate((Torso_centered_xz[[0,1,3,2],x],Torso_centered_xz[[0],x])), np.concatenate((Torso_centered_xz[[0,1,3,2],y],Torso_centered_xz[[0],y])),np.zeros((5)), color='y')

            # ax2.scatter(new_coordinate[x,0], new_coordinate[y,0], 0,  marker='o',color='r')
            # ax2.scatter(new_coordinate[x,1], new_coordinate[y,1], 0,  marker='o',color='m')
            # ax2.scatter(new_coordinate[x,2], new_coordinate[y,2], 0,  marker='o',color='b')
            # ax2.scatter(new_coordinate[x,3], new_coordinate[y,3], 0,  marker='o',color='c')
            # ax2.plot([0,0],[0,0.1],color='r')
            # ax2.plot([0,0.1],[0,0],color='r')

            # ax2.scatter(Torso_centered_xz[0,x], Torso_centered_xz[0,y],0,  marker='x',color='r')
            # ax2.scatter(Torso_centered_xz[1,x], Torso_centered_xz[1,y],0,  marker='x',color='m')
            # ax2.scatter(Torso_centered_xz[2,x], Torso_centered_xz[2,y],0,  marker='x',color='b')
            # ax2.scatter(Torso_centered_xz[3,x], Torso_centered_xz[3,y],0,  marker='x',color='c')
            # ax2.set_xlabel('X [m]')
            # ax2.set_ylabel('Z [m]')
            # ax2.set_zlabel('Y [m]')
            # ax2.view_init(azim=145, elev=-140)
            # plt.show()
            # plt.savefig(f"Frame_fig/Torso_frame_{frame_index}.png")

             # # List to save your projections to
            # projections = []
            # # This is called everytime you release the mouse button
            # def on_click(event):
            #     azim, elev = ax2.azim, ax2.elev
            #     projections.append((azim, elev))
            #     print('azimuth',azim,'elevation', elev)

            # cid =fig.canvas.mpl_connect('button_release_event', on_click)
            #plt.show()"""
        
        end_rotation= time.perf_counter()
        start_plot= time.perf_counter()

        if args.plot_skeleton  :
            if args.plot_real_time:
                plt.close()
                plt.ion()
            fig = plt.figure(figsize=(20.0, 10.0))
            ax1 = fig.add_subplot(121)
            ax1.imshow(color_plt)
            ax2 = fig.add_subplot(122,projection='3d')
            for people_cartesian_pred in cartesian_pred:
                for pair in pairs:
                    ax2.plot(people_cartesian_pred[0,pair], people_cartesian_pred[1,pair], people_cartesian_pred[2,pair])
            ax2.plot([0,0.1],[0,0],[0,0],c='b')
            ax2.plot([0,0],[0,0.1],[0,0],c='g')
            ax2.plot([0,0],[0,0],[0,0.1],c='r')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_xlim3d(-1.5, 1.5)
            ax2.set_ylim3d(-1, 1)
            ax2.set_zlim3d(-1, 1)
            #front view
            ax2.view_init(azim=88, elev=-68)
            #up view
            #ax2.view_init(azim=88, elev=-2)

            #if args.plot_real_time:
                #plt.draw()
                #plt.pause(0.001)
            
            
            plt.savefig(f"Frame_fig/{frame_index}.png")
            plt.show()
            if args.plot_real_time:
                plt.ioff()

        if args.plot_pointcloud:

            color_img=o3d.geometry.Image(color_plt)
            depth_img=o3d.geometry.Image(np.copy(depth_map*1000).astype(np.uint16))
            rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color_img,depth_img,depth_scale=1000,depth_trunc=8,convert_rgb_to_intensity=False)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(width,heigth,fx,fy,cx,cy))
                            
            if args.project_pointcloud_torso_frame is False :
                # flip it, otherwise the pointcloud will be upside down
                pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


            if args.plot_real_time==False:
                #o3d.visualization.draw_geometries([rgbd_image])
                mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)#.scale(0.4, center=(0, 0, 0))

                
                #Visualize Point Cloud
                vis.add_geometry(mesh)
                vis.add_geometry(pcd)
                #view control
                ctr = vis.get_view_control()
                #Viewpoint in torso frame
                if args.project_pointcloud_torso_frame:
                    #From neural networks
                    # ctr.set_front(np.array( [0.5322687991822731, -0.12341988666161863, -0.83753057078144577] ) )
                    # ctr.set_lookat(np.array( [ -1.8111004207019175, 0.82685655683616077, 1.7260878402305799 ]) )
                    # ctr.set_up(np.array( [ 0.030590938857055654, 0.99147386266430093, -0.12666402059533283 ]) )
                    # ctr.set_zoom(0.619)

                    ctr.set_front([0.48397772265932698, 0.12982095147818667, -0.86539706755153145])
                    ctr.set_lookat([ -1.7501399058158631, -0.51045015102172453, 3.3384999942779543])
                    ctr.set_up([ -0.018336707055478629, -0.98721279496109449, -0.15834981098651696  ])
                    ctr.set_zoom(0.3)

                    if args.project_pointcloud_torso_frame:
                        mesh.rotate(R,center=(0, 0, 0))
                        mesh.translate(Torso_centroid,relative=True)
                    # ctr.set_front(np.array( [-0.98968588642431865, -0.12829040962206251, -0.063744937144168651] ) )
                    # ctr.set_lookat(np.array( [0.76388329296391633, 0.64673637370853398, 0.57562247451868942]) )
                    # ctr.set_up(np.array( [ -0.12710706045686668, 0.99163825741717204, -0.022301605517290313 ]) )
                    # ctr.set_zoom(0.49999999999999978)
                # viewpoint in camera frame
                else:
                    ctr.set_front([-0.23210059274685624, 0.044402826702813564, 0.97167777777787945])
                    ctr.set_lookat([ 1.7687440243258228, 0.77902163585813933, -2.292000102996826 ])
                    ctr.set_up([  0.016681922319671701, 0.99899230957010909, -0.041666279981538441  ])
                    ctr.set_zoom(0.3)

                
                # Updates
                vis.update_geometry(pcd)
                vis.update_geometry(mesh)
                vis.poll_events()
                vis.update_renderer()
                #vis.run()
                # Capture image
                vis.capture_screen_image("Frame_fig/{}.png".format(frame_index))
                #image = vis.capture_screen_float_buffer()
                # Remove previous geometry
                vis.remove_geometry(pcd)
                vis.remove_geometry(mesh)

                print('plotted')
                # #mesh_RT = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate((R), center=(1,1,1))
                # #mesh_RT.translate(-T, relative=True)#, center=(T[0],T[1],T[2]))
                #o3d.visualization.draw_geometries([mesh,pcd])

        frame_index+=1
        end_plot= time.perf_counter()

        end_callback= time.perf_counter()

        depth_computing_time=end_depthmap-start_depthmap
        open_computing_time=end_open-start_open
        rotation_computing_time=end_rotation-start_rotation
        plot_computing_time=end_plot-start_plot
        total_computing_time=end_callback-start_callback

        print('-----------------------------------------------------------------------------')
        print('Total time', int((total_computing_time)*1000),'ms thus ',1/(total_computing_time),'fps/Hz')
        print('Openpifpaf time', int((open_computing_time)*1000),'ms thus ',1/(open_computing_time),'fps/Hz')
        print('Time to rotate from camera frame to torso frame', int((rotation_computing_time)*1000),'ms thus ',1/(rotation_computing_time),'fps/Hz')
        print('Plotting time', int((plot_computing_time)*1000),'ms thus ',1/(plot_computing_time),'fps/Hz')
        print('Depth time', int((depth_computing_time)*1000),'ms thus ',1/(depth_computing_time),'fps/Hz')
        print('-----------------------------------------------------------------------------')


















_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def create_cloud(header, fields, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message.
    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param fields: The point cloud fields.
    @type  fields: iterable of L{sensor_msgs.msg.PointField}
    @param points: The point cloud points.
    @type  points: list of iterables, i.e. one iterable for each point, with the
                   elements of each iterable being the values of the fields for 
                   that point (in the same order as the fields parameter)
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))

    buff = ctypes.create_string_buffer(cloud_struct.size * len(points))

    point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
    offset = 0
    for p in points:

        pack_into(buff, offset, *p)
        offset += point_step
    #print('cloud_struct.size',cloud_struct.size)
    #print('width',len(points))
    return PointCloud2(header=header,
                       height=1,
                       width=len(points),
                       is_dense=False,
                       is_bigendian=False,
                       fields=fields,
                       point_step=cloud_struct.size,
                       row_step=cloud_struct.size * len(points),
                       data=buff.raw)

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

def rigid_transform_3D(A, B):
    """
    Input: expects 3xN matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector"""
    #Credit to http://nghiaho.com/?page_id=671
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if np.linalg.matrix_rank(H) < 3:
    #     raise ValueError("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < 0, reflection detected!, correcting for it ...")
        Vt[1,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
#------------------------------------------------------------------------------------------------------------------


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
        #print("device: %s" % device)

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
            #print(f"args.model_type '{args.model_type}' not implemented, use: --args.model_type large")
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

if __name__ == '__main__':
    main()