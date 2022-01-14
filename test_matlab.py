import scipy.io
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('../video_test/depth/a24_s1_t1_depth.mat')

f = open('json1_1.json',)
 
depth_GT=mat["d_depth"]
depth_GT=cv2.resize(depth_GT,(640,480))
RMS=[]
abs_rel=[]
data = []
for idx,line in enumerate(open('json1_1.json', 'r')):
    data.append(json.loads(line)) 
    GT_keypoints=[]
    keypoints=np.reshape((data[idx]["predictions"][0]["keypoints"]),(-1,3)).astype(int)
    depth_pred=np.array((data[idx]["depth_pred"]))
    RMS_temp=[]
    abs_rel_tmp=[]
    for i,keypoint in enumerate(keypoints):
        #print(keypoint)
        #print(depth_GT[keypoint[1],keypoint[0],idx]/1000)
        depth_GT[keypoint[1],keypoint[0],idx]=1000
        GT_keypoints.append
        RMS_temp.append(np.sqrt(np.square(depth_GT[keypoint[1],keypoint[0],idx]/1000-depth_pred[idx,i])))
        abs_rel_tmp.append((depth_GT[keypoint[1],keypoint[0],idx]/1000-depth_pred[idx,i])/(depth_GT[keypoint[1],keypoint[0],idx]/1000))
    RMS.append(np.nanmean(RMS_temp))
    abs_rel.append(np.nanmean(abs_rel_tmp))
    print('rms',RMS)
    plt.imshow(depth_GT[:,:,idx])
    plt.show()
    break
    #print(idx,'\n',keypoints)

#print(mat["d_depth"])