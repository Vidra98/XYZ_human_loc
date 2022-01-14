# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model
import os
import cv2
import numpy as np
import torch.autograd as autograd
import time
from skimage import transform


BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

input = 'test_data/exosquellete.mp4'

eval_num_threads = 2

model = pix2pix_model.Pix2PixModel(opt)

compared_img=False
show=False
Save_video=False

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0

print(
    '=================================  BEGIN VALIDATION ====================================='
)

print('TESTING ON VIDEO')

height_model = 288
width_model = 512

resized_height=360
resized_width=640

cap = cv2.VideoCapture(input)
count = 0
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")
    print(input)

if Save_video==True:
    #Parameters of output video    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mp4',fourcc, 20, (resized_width,resized_height))

#Video processing
    
while cap.isOpened():
    ret,frame = cap.read()
    if ret==True:

        #resize for the model
        frame=torch.tensor(frame)

        frame = np.float32(frame)/255.0
        frame = transform.resize(frame, (width_model,height_model))
        frame=torch.tensor(frame)

        frame=torch.transpose(frame,0,2)
        frame=torch.transpose(frame,1,2).unsqueeze(0)

        input_imgs = autograd.Variable(frame, requires_grad=False)

        #run the model
        start = time.time()
        prediction_d, pred_confidence = model.netG.forward(frame)
        #print(prediction_d)
        pred_log_d = prediction_d.squeeze(1)
        pred_d = torch.exp(pred_log_d)
        end = time.time()
        print("Time: " + str(end - start)+", FPS :"+str(1/(end - start)))

        saved_img = np.transpose(
            input_imgs[0, :, :, :].cpu().numpy(), (1, 2, 0))

        pred_d_ref = pred_d.data[0, :, :].cpu().numpy()
        disparity = 1. / pred_d_ref

        disparity = disparity / np.max(disparity)
        disparity = np.tile(np.expand_dims(disparity, axis=-1), (1, 1, 3))

        if compared_img==True:
            saved_imgs = np.concatenate((saved_img, disparity), axis=1)
        else :
            saved_imgs=disparity

        saved_imgs = (saved_imgs*255).astype(np.uint8)
        saved_imgs = cv2.resize(saved_imgs, (resized_width, resized_height))

        if Save_video==True:
            out.write(saved_imgs)
        
        if show==True:
            cv2.imshow('depth map',saved_imgs)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count = count + 1

cap.release()
out.release()
cv2.destroyAllWindows() # destroy all opened windows  
