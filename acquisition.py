import cv2
import pyrealsense2 as rs
import json
import numpy as np
import time

#---------------------------------------Parameters ---------------------------------------------------
acquisition_time=10 #in second
show_acquisition=True
frame_rate=30.0
data_nb=0

show_input=True
show_depth=True
#-----------------------------------------------------------------------------------------------------


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


point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera() 

#Constant
ascii_esc=27
ascii_tab=9
ascii_s=115
ascii_d=100

#Path
storing_file='input/'

#Acquisition variable
set_acquisition=0

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

res_640=(640,480)
rw=640
rh=360
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#depth_out = cv2.VideoWriter('test.mp4',fourcc, frame_rate, res_640)
frame=0
save_dict={}
while True:
    ret, depth_frame, color_frame = dc.get_frame()
    # Show input
    if show_input==True:
        show_frame=np.copy(color_frame)
        cv2.circle(show_frame, point, 4, (0, 0, 255))
        distance = depth_frame[point[1], point[0]]/1000
        cv2.putText(show_frame, "{}m".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow("Color frame", show_frame)
    #show depth map
    if show_depth==True:
        show_depth_map=cv2.applyColorMap(255-(np.array(depth_frame)/np.max(depth_frame)*255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        cv2.imshow("depth map", show_depth_map)

    #Store video
    if set_acquisition==True :
        color_frame = cv2.resize(color_frame, res_640)
        video_out.write(color_frame)
        save_dict["frame{}".format(frame)]=depth_frame.tolist()#[1:3,6:9].tolist()
    #depth_out.write(depth_frame)
    key = cv2.waitKey(1)
    #Write in hard drive
    if set_acquisition==True and frame==acquisition_time*frame_rate:
        frame=0
        set_acquisition=False
        video_out.release()
        with open(storing_file+'data{}.json'.format(data_nb), 'w') as video_outfile:
            t0= time.time()
            json.dump(save_dict, video_outfile)
            t1= time.time()
            print('writing took {} seconds for video of {} seconds'.format(t1-t0,acquisition_time))

    #Launch acquisition
    if key==ascii_s:
        data_nb=data_nb+1
        video_out = cv2.VideoWriter(storing_file+'output{}.mp4'.format(data_nb),fourcc, frame_rate, res_640)
        frame=0
        set_acquisition=True
        print('{} will be saved in'.format(frame)+storing_file+'data{}.json \n'.format(data_nb) )

    if key == ascii_esc:
        video_out.release()
        cv2.destroyAllWindows()
        break

    if key == ascii_d:
        print('depth_frame dimension',np.shape(depth_frame))
        print('video output dim was 630*360')


    if set_acquisition==True : frame=frame+1 
    
