# import cv2
# import numpy as np
# width=512
# height=256
# saved_imgs=200*np.ones((width,height)).astype(np.uint8)
# cv2.imshow('huh',saved_imgs)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width,height))

# count=0
# while(count<200):
#     count=count+1
#     saved_imgs = cv2.resize(saved_imgs, (width,height))

#     out.write(saved_imgs)

#     #Video processing
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# out.release()
# cv2.destroyAllWindows() # destroy all opened windows    
import numpy as np
import cv2

cap = cv2.VideoCapture('test_data/exosquellete.mp4')
rw=640
rh=360
(grabbed, frame) = cap.read()
# fshape = frame.shape
# fheight = fshape[0]
# fwidth = fshape[1]
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('test.mp4',fourcc, 30.0, (rw,rh))

#print(fwidth,fheight)
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(rw), rh))

    #if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
    out.write(frame)

    #    cv2.imshow('frame',frame)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    #else:
    #    break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()