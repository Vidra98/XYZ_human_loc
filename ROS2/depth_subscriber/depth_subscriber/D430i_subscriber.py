from numpy.core.numeric import NaN
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
from depth.mannequin.models import pix2pix_model
import matplotlib.pyplot as plt

from torchvision.transforms import Compose

from depth.midas import midas_cli
from depth.midas.midas.dpt_depth import DPTDepthModel
from depth.midas.midas.midas_net import MidasNet
from depth.midas.midas.midas_net_custom import MidasNet_small
from depth.midas.midas.transforms import Resize, NormalizeImage, PrepareForNet

import open3d as o3d


# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1)] #PointField.UINT32 TO_DEL

# Bit operations
BIT_MOVE_24 = 2**24
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8

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


class D430i_publisher(Node):

    def __init__(self):
        super().__init__('pcd_publisher_node')
        
        # I create a publisher that publishes sensor_msgs.PointCloud2 to the 
        # topic 'self.pcd'. The value '10' refers to the history_depth, which I 
        # believe is related to the ROS1 concept of queue size. 
        # Read more here: 
        # http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.pcd_publisher = self.create_publisher(PointCloud2, '/semester_project/camera_color_optical_frame', 1)

    def callback(self,depth_pcd):
        self.pcd_publisher.publish(depth_pcd)




class D430i_suscriber(Node):

    def __init__(self,args,predictor,capture,depth_model,transform,publisher,vis):
        super().__init__('D430i_suscriber')

        # delay (in seconds) with which messages can be synchronized.
        self.slop=0.005
        #how many sets of messages it should store from each input filter (by timestamp) while waiting for messages to arrive and complete their “set”
        self.queue_size=1

        self.color_sub = message_filters.Subscriber(self, Image,"/camera/color/image_raw")
        self.depth_sub = message_filters.Subscriber(self, Image,"/camera/aligned_depth_to_color/image_raw")
        self.tss = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub],
                                                               queue_size=self.queue_size, slop=self.slop)
        self.tss.registerCallback(self.listener_callback)

        self.clock = rclpy.clock.Clock()

        self.vis=vis

        self.args=args
        self.predictor=predictor
        self.depth_model=depth_model
        self.transform=transform
        self.capture=capture
        self.publisher=publisher
        self.frame_index=0

    def listener_callback(self, color_msg,depth_msg):
        torch.cuda.empty_cache()
        color_plt = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
        depth_plt = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width, -1)
        start_callback = time.perf_counter()
        
        image, processed_image, anns, meta = self.capture.preprocessing(torch.tensor(color_plt))

        #We resize the input in openpifpaf networks input dimension.
        cap=[[image,torch.tensor(np.expand_dims(processed_image.numpy(),0)), [[]], np.expand_dims(meta,0)]]#(np.expand_dims(anns,0)

        preds=self.predictor.dataloader(cap)
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

        start_depthmap = time.perf_counter()
        if self.args.GT_from_camera is False:
            depth_map, disparity, _=depth_mapping(self.depth_model,self.transform,color_plt,self.args)
            fig3, ax3 = plt.subplots()
            im3 = ax3.imshow(depth_map,cmap='gray')
        elif self.args.GT_from_camera :
            depth_map=depth_plt/1000
        
        #The scaling and shifting uses GT:
        if self.args.shift_scale and self.args.GT_from_camera is False :
            #We need the GT either for outputting a GT map or a doing the shifting and scaling
            depth_GT = depth_plt[:,:,0]
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

        end_callback= time.perf_counter()

        depth_computing_time=end_depthmap-start_depthmap
        total_computing_time=end_callback-start_callback

        print('-----------------------------------------------------------------------------')
        print('Total time', int((total_computing_time)*1000),'ms thus ',1/(total_computing_time),'fps/Hz')
        print('Openpifpaf time', int((total_computing_time-depth_computing_time)*1000),'ms thus ',1/(total_computing_time-depth_computing_time),'fps/Hz')
        print('Depth time', int((depth_computing_time)*1000),'ms thus ',1/(depth_computing_time),'fps/Hz')
        print('-----------------------------------------------------------------------------')
        width,heigth,_=np.shape(color_plt)

        #print(len(preds_indice))
        if len(preds_indice)>0:
            for kp in preds_indice[0,:]:
                if any(np.array(kp)==0) :
                    continue
                else:
                    color_plt=cv2.circle(color_plt,tuple([kp[1],kp[0]]), 10, (255,0,0),-1) 
              
        
        #These are the intrisic of the camera (in pxl), they can be obtain on the P array of the camera info topics
        #http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        fx=909.2
        fy=909.0
        cx=653.7
        cy=367.6
        s=0
        color_plt=cv2.circle(color_plt,tuple([int(cx),int(cy)]), 10, (0,255,0),-1) 
        

        #For each person we predict the depth
        depth_pred=np.zeros((len(preds_indice),17))
        for people_idx,people_keypoints in enumerate(preds_indice) :
            for bodypart_idx,bodypart_keypoints in enumerate(people_keypoints) :
                depth_pred[people_idx,bodypart_idx]=depth_map[tuple(bodypart_keypoints)]
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
            print('shoulder shoulder dist',np.sqrt((cartesian_pred[0,0,left_shoulder]-cartesian_pred[0,0,right_shoulder])**2+(cartesian_pred[0,1,left_shoulder]-cartesian_pred[0,1,right_shoulder])**2))
            print('hips hips dist',np.sqrt((cartesian_pred[0,0,left_hip]-cartesian_pred[0,0,right_hip])**2+(cartesian_pred[0,1,left_hip]-cartesian_pred[0,1,right_hip])**2))
            print('left shoulder hips dist',np.sqrt((cartesian_pred[0,0,left_shoulder]-cartesian_pred[0,0,left_hip])**2+(cartesian_pred[0,1,left_shoulder]-cartesian_pred[0,1,left_hip])**2))
            print('right shoulder hips dist',np.sqrt((cartesian_pred[0,0,right_shoulder]-cartesian_pred[0,0,right_hip])**2+(cartesian_pred[0,1,right_shoulder]-cartesian_pred[0,1,right_hip])**2))

            #We want to compute body orientation in the body reference frame
            Torso_coordinate=np.array([cartesian_pred[0,:,left_shoulder],cartesian_pred[0,:,right_shoulder],cartesian_pred[0,:,left_hip],cartesian_pred[0,:,right_hip]])
            Torso_coordinate=Torso_coordinate[~np.isnan(Torso_coordinate)]
            Torso_coordinate=np.reshape(Torso_coordinate,(int(len(Torso_coordinate)/3),3))
        else :
            Torso_coordinate=[]

        #We need at least three keypoint of the torso to do a projection on the torso frame
        if len(Torso_coordinate)<3:
            print('At least three point need to be located on the torso to project to the torso frame - Camera frame is used.')
            R=np.eye(3)
            T=np.array([0,0,0])
            Torso_centroid=np.array([0,0,0])
        else :
            #Center of the new frame
            Torso_centroid=np.mean(Torso_coordinate,0)

            #We shearch the principal axis for the xz plan, we want to keep y pointb up in camera frame 
            Torso_centered_xz=Torso_coordinate[:,[0,2]]-Torso_centroid[[0,2]]
            Torso_variance_xz=np.dot((Torso_centered_xz).transpose(),(Torso_centered_xz))/len(Torso_centered_xz)
        

            #The eigenvector of the covariance matrix, gives us the pricipals axis where we will maximize information. Meaning the axis align with the body 
            eigen_value,eigenvector=np.linalg.eig(Torso_variance_xz)

            #On X,Z it's the projection of the point on the new referentiel, on the Y axis we just invert the axis.
            new_coordinate=np.dot(eigenvector.transpose(),Torso_centered_xz.transpose())
            
            R,T=rigid_transform_3D(np.array(Torso_coordinate.transpose()[[0,2]]),new_coordinate)

            R_tmp=np.zeros((3,3))
            R_tmp[tuple([[0,0,2,2],[0,2,0,2]])]=np.reshape(R,(1,4))
            R_tmp[1,1]=-1
            R=R_tmp
            T=np.array([[T.item(0)],[Torso_centroid[1]],[T.item(1)]])

            cartesian_pred=R@(cartesian_pred-np.array([[Torso_centroid[0]],[Torso_centroid[1]],[Torso_centroid[2]]]))

        if self.args.plot_skeleton  :
            if self.args.plot_real_time:
                plt.close()
                plt.ion()
            fig = plt.figure(figsize=(20, 10))
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
            #front view
            ax2.view_init(azim=88, elev=-68)
            #up view
            #ax2.view_init(azim=88, elev=-2)

            if self.args.plot_real_time:
                plt.draw()
                plt.pause(0.001)
            

            # # List to save your projections to
            # projections = []
            # # This is called everytime you release the mouse button
            # def on_click(event):
            #     azim, elev = ax2.azim, ax2.elev
            #     projections.append((azim, elev))
            #     print('azimuth',azim,'elevation', elev)

            # cid =fig.canvas.mpl_connect('button_release_event', on_click)
            self.frame_index+=1
            plt.savefig(f"Frame_fig/{self.frame_index}.png")
            #plt.show()
            if self.args.plot_real_time:
                plt.ioff()

        color_img=o3d.geometry.Image(color_plt)
        depth_img=o3d.geometry.Image(np.copy(depth_map*1000).astype(np.uint16))
        rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color_img,depth_img,depth_scale=1000,depth_trunc=8,convert_rgb_to_intensity=False)

        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(width,heigth,fx,fy,cx,cy))
        # flip it, otherwise the pointcloud will be upside down
        #self.pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #Uncomment to project the pointcloud
        self.pcd.rotate(R,center=(0,0,0))
        self.pcd.translate(T)
        #self.pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        if self.args.plot_real_time==False and self.args.plot_pointcloud:
            #o3d.visualization.draw_geometries([rgbd_image])
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()#.scale(0.4, center=(0, 0, 0))
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)
            vis.add_geometry(self.pcd)
            
            ctr = vis.get_view_control()
            ctr.set_front(np.array( [ -0.35217120750063657, -0.075318856934746256, -0.93290005380940288 ] ) )
            ctr.set_up(np.array( [ -0.0050671332772632876, 0.99689550387908976, -0.078572759312667176 ]) )
            ctr.set_lookat(np.array( [ 1.2204758770112838, 1.1086807226732882, 2.6767072071018365 ]) )
            ctr.set_zoom(0.5)
            vis.run()

        elif self.args.plot_real_time and self.args.plot_pointcloud :
            #Since update geometry doesn't works in our case we use a roundaround way to avoid the issue, subtoptimal.
            # self.vis.update_geometry(self.pcd)
            self.vis.clear_geometries()
            self.vis.add_geometry(self.pcd)
            ctr = self.vis.get_view_control()
            ctr.set_front(np.array( [ -0.35217120750063657, -0.075318856934746256, -0.93290005380940288 ] ) )
            ctr.set_up(np.array( [ -0.0050671332772632876, 0.99689550387908976, -0.078572759312667176 ]) )
            ctr.set_lookat(np.array( [ 1.2204758770112838, 1.1086807226732882, 2.6767072071018365 ]) )
            self.vis.poll_events()
            self.vis.update_renderer()
        
        if self.args.publish_pointcloud:
            # I convert point cloud to array to transfer it to ros
            points = np.asarray(self.pcd.points)

            # Set "header"
            header = Header()
            header.stamp = self.clock.now().to_msg()
            header.frame_id = "map"

            # Set "fields" and "cloud_data"
            points=np.asarray(self.pcd.points)
            if not self.pcd.colors: # XYZ only
                fields=FIELDS_XYZ
                cloud_data=points
            else: # XYZ + RGB
                fields=FIELDS_XYZRGB
                # -- Change rgb color from "three float" to "one 24-byte int"
                # 0x00FFFFFF is white, 0x00000000 is black.
                colors = np.floor(np.asarray(self.pcd.colors)*255) # nx3 matrix
                colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
                cloud_data=np.c_[points, colors]

            # create ros_cloud
            pcd_to_publish=create_cloud(header, fields, cloud_data.astype(int))
            
            self.publisher.callback(pcd_to_publish)
        return

def main(args=None):
    rclpy.init(args=args)
    torch.cuda.empty_cache()
    args = cli()
    args.checkpoint='mobilenetv2'
    args.shift_scale=True
    args.model_weigth='dpt_hybrid' # Midas model dpt_hybrid, midas_v21_large or midas_v21_small'
    # Set true for real time plot and false for single plot a time
    args.publish_pointcloud=False
    args.plot_pointcloud=True
    args.plot_real_time=False

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
    if args.plot_real_time==True and args.plot_pointcloud:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd=o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
    else :
        vis=None

    publisher=D430i_publisher()
    minimal_subscriber = D430i_suscriber(args,predictor,capture,depth_model,transform,publisher,vis)
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

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
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
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
        frame = cv2.resize(frame, (width_depthmap,height_depthmap))

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