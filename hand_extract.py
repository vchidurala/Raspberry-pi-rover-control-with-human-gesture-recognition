import imutils
import dlib
import cv2
import numpy as np
import argparse
from imutils import face_utils
import faceBlendCommon as fbc
import os
import tensorflow as tf, sys

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
leftEye = [36, 37, 38, 39, 40, 41]
rightEye = [42, 43, 44, 45, 46, 47]
mouth = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 ]
leftBrows = [17, 18, 19, 20, 21]
rightBrows = [22, 23, 24, 25, 26]

# kernal size for morphological opening 
k = 5

def applyMask(skinImage, points):

  tempMask = np.ones((skinImage.shape[0], skinImage.shape[1]), dtype = np.uint8)
  
  temp = []
  for p in leftEye:
    temp.append(( points[p][0], points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in rightEye:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in leftBrows:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in rightBrows:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in mouth:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  return cv2.bitwise_and(skinImage, skinImage, mask = tempMask)

def findSkinYCB(meanimg, frame):

  # Specify the offset around the mean value
  CrOffset = 15
  CbOffset = 15
  YValOffset = 100
  
  # Convert to the YCrCb color space
  ycb = cv2.cvtColor(meanimg,cv2.COLOR_BGR2YCrCb)[0][0]
  frameYCB = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)

  # Find the range of pixel values to be taken as skin region
  minYCB = np.array([ycb[0] - YValOffset,ycb[1] - CrOffset, ycb[2] - CbOffset])
  maxYCB = np.array([ycb[0] + YValOffset,ycb[1] + CrOffset, ycb[2] + CbOffset])

  # Apply the range function to find the pixel values in the specific range
  skinRegionycb = cv2.inRange(frameYCB,minYCB,maxYCB)

  # Apply Gaussian blur to remove noise
  skinRegionycb = cv2.GaussianBlur(skinRegionycb, (5, 5), 0)

  # Get the kernel for performing morphological opening operation
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
  skinRegionycb = cv2.morphologyEx(skinRegionycb, cv2.MORPH_OPEN, kernel, iterations = 3)
  #skinRegionycb = cv2.dilate(skinRegionycb, kernel, iterations=3)

  # Apply the mask to the image
  skinycb = cv2.bitwise_and(frame, frame, mask = skinRegionycb)
  return skinRegionycb,skinycb

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
def handconvex(bin_img,img):
    _,contours, hierarchy = cv2.findContours(bin_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape,np.uint8)
    max_area=0
    for i in range(len(contours)):
      cnt=contours[i]
      area = cv2.contourArea(cnt)
      if(area>max_area):
        max_area=area
        ci=i

    cnt=contours[ci]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00']!=0:
      cx = int(moments['m10']/moments['m00']) # cx = M10/M00
      cy = int(moments['m01']/moments['m00']) # cy = M01/M00
    centr=(cx,cy)       
    cv2.circle(img,centr,5,[0,0,255],2)       
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2) 
    cv2.drawContours(drawing,[hull],0,(0,0,255),2) 
          
    cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    hull = cv2.convexHull(cnt,returnPoints = False)

    return drawing,x,y,w,h
def classifyhand(image_data):
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    label_lines = [line.rstrip() for line 
                     in tf.gfile.GFile("retrained_labels_massay_mobile.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph_massay_mobile.pb", 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='') 
    with tf.Session() as sess:
      # Feed the image_data as input to the graph and get first prediction
      softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
      predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg:0': image_data})
            
      # Sort to show labels of first prediction in order of confidence
      top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
      pf=[]
      pfs=[]
      for node_id in top_k:
        human_string = label_lines[node_id]
        pf.append(human_string)
        score = predictions[0][node_id]
        pfs.append(score)
        print('%s (score = %.5f)' % (human_string, score))
      print('-------------------------------------------')
      text=str(pf[0])
      #text=str(pf[0])+'(Confidance: '+str(pfs[0])+')' 
    return text
def distance(x,y):
    import math
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2) 




  

if __name__ == '__main__':

    
    print("[INFO] loading facial landmark predictor...")
    #faceDetector = dlib.get_frontal_face_detector()
    detector = dlib.get_frontal_face_detector()
    # Load landmark detector.
    #landmarkDetector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    print("[INFO] camera sensor warming up...")

    camera = cv2.VideoCapture(0)

    # keep looping over the frames in the video
fr=1;
im=1;
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    size = frame.shape
    #print(frame[479][848][0])
    cv2.circle(gray,(0,0),100,(0,0,255),-1)

    # Camera internals
     
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    if len(rects) != 0:
        maxArea = 0
        maxRect = None
        # TODO: test on images with multiple faces
        for rect in rects:
            if rect.area() > maxArea:
                maxArea = rect.area()
                maxRect = [rect.left(),rect.top(),rect.right(),rect.bottom()]
        
        rect = dlib.rectangle(*maxRect)
        shape = predictor(gray, rect)
        
        landmarks = fbc.dlibLandmarksToPoints(shape)
        landmarks = np.array(landmarks)
        
        ix = landmarks[32][0]
        fx = landmarks[34][0]
        iy = landmarks[29][1]
        fy = landmarks[30][1]

        # Take a patch on the nose
        tempimg = frame[iy:fy,ix:fx,:]
      
        # Compute the mean image from the patch
        meanimg = np.uint8([[cv2.mean(tempimg)[:3]]])
        skinRegionycb,skinycb = findSkinYCB(meanimg, frame)


        maskedskinycb = applyMask(skinycb, landmarks)
        #cv2.putText(skinycb, "YCrCb", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow('masked',maskedskinycb)
        cv2.imshow("YCrCb",skinRegionycb)
        x1=int(landmarks[0][0])
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            #cv2.circle(frame, (400,200),10, (255,0,0),-1)
  
        newframe=frame[:,0:x1]
        maskedframe=maskedskinycb[:,0:x1]
        handregion=skinRegionycb[:,0:x1]
        drawing,xh,yh,wh,hh=handconvex(handregion,newframe)
        #print("[INFO] Extracting the Hand region...")
        cv2.imshow('output',drawing)
        #crphand=newframe[yh:yh+hh,xh:xh+wh]
        crphand=maskedframe[yh:yh+hh,xh:xh+wh]
        #cv2.imshow('crophand',crphand)
        

        ## Resizeing the image ##
        r = 90.0 / crphand.shape[1]
        dim = (90, int(crphand.shape[0] * r))
        # perform the actual resizing of the image and show it
        #resized = cv2.resize(crphand, dim, interpolation = cv2.INTER_AREA)
        resized = crphand
        #cv2.imshow("resized", resized)
        #cv2.imshow("hand",newframe)
        opdir='/finger/op_imgs/'
        img_files=os.listdir(opdir)
        len(img_files)
        if len(img_files)<2000:
        #  #print("[INFO] Capturing the frames for classification...")
          if fr==3:
            cv2.imwrite(opdir+"hand-" + str(im) + ".jpg", resized)
        #    #print("[INFO] classifying...")
        #    #text=classifyhand(resized)
            fr=1

        fr=fr+1


        
        font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Display image
        im=im+1
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# cleanup the camera and close any open windows

camera.release()
cv2.destroyAllWindows()
