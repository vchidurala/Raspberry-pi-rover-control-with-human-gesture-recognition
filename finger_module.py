from __future__ import print_function
from google.assistant.library import Assistant
from google.assistant.library.event import EventType
from google.assistant.library.file_helpers import existing_file
from scripts.label_image  import *
import numpy as np
import cv2
import tensorflow as tf
import scipy.misc
import time
import requests
import argparse
import os.path
import json
import time 
import requests
import google.oauth2.credentials
import datetime
import google.assistant.library.assistant as gala
import dlib
import faceBlendCommon as fbc

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

    #if(1):
    #  defects = cv2.convexityDefects(cnt,hull)
    #  #print(defects)
    #  print(defects.ndim)
    #  mind=0
    #  maxd=0
    #  if defects.all():
    #    for i in range(defects.shape[0]):
    #      s,e,f,d = defects[i,0]
    #      start = tuple(cnt[s][0])
    #      end = tuple(cnt[e][0])
    #      far = tuple(cnt[f][0])
    #      dist = cv2.pointPolygonTest(cnt,centr,True)
    #      cv2.line(img,start,end,[0,255,0],2)
    #      cv2.circle(img,far,5,[0,0,255],-1)
    #    i=0
    return drawing,x,y,w,h
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
        input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def classifyhand(file_name):
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    model_file = "/finger/directions/retrained_graph.pb"
    label_file = "/finger/directions/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"
    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    pf=[]
    pfs=[]
    for i in top_k:
      pf.append(labels[i])
      pfs.append(results[i])
      print('%s (score = %.5f)' % (labels[i], results[i]))
    print('-------------------------------------------')
    text=str(pf[0])
    #text=str(pf[0])+'(Confidance: '+str(pfs[0])+')' 
    return text

flag =1
###process event definition ####

def process_event(event):
    """Pretty prints events.

    Prints all events that occur with two spaces between each new
    conversation and a single space between turns of a conversation.

    Args:
        event(event.Event): The current event to process.
    """
    if event.type == EventType.ON_CONVERSATION_TURN_STARTED:
        print()

    print(event)

    if event.type == EventType.ON_RECOGNIZING_SPEECH_FINISHED:
        speech_text = event.args["text"]
        print(" Going to send the speech text to the RPi: " + speech_text)	
        r=requests.post("http://IP:Port", data ={'direction': speech_text})
        print(" sent the speech text to the RPi: " + speech_text)

    if (event.type == EventType.ON_CONVERSATION_TURN_FINISHED and
            event.args and not event.args['with_follow_on_turn']):
        print()


print("[INFO] loading facial landmark predictor...")
#faceDetector = dlib.get_frontal_face_detector()
detector = dlib.get_frontal_face_detector()
# Load landmark detector.
#landmarkDetector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
print("[INFO] camera sensor warming up...")

##finger module ##
cap = cv2.VideoCapture(0)

input_height=224
input_width=224			
input_mean=0 
input_std=255

model_file = "directions/retrained_graph.pb" 
label_file =  "directions/retrained_labels.txt"
img = cv2.imread('batss.jpg',0)
input_layer = "input"
output_layer = "final_result"
##sing it module##
parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--credentials', type=existing_file,
                        metavar='OAUTH2_CREDENTIALS_FILE',
                        default=os.path.join(
                            os.path.expanduser('~/.config'),
                            'google-oauthlib-tool',
                            'credentials.json'
                        ),
                        help='Path to store and read OAuth2 credentials')
args = parser.parse_args()
with open(args.credentials, 'r') as f:
        credentials = google.oauth2.credentials.Credentials(token=None,
                                                            **json.load(f))
print(credentials)
#    print(time.clock())
cdt = datetime.datetime.now()
print(cdt.minute)	

print(flag)
print('lets begin..')
fr=1
im=1
while(1):
	ret, frame = cap.read()
	scipy.misc.imsave('outfile.jpg', frame)
	file_name= 'outfile.jpg'
################ finger module flag ==1 ################
	if cv2.waitKey(1) & 0xFF == ord('s') or flag == 1:
		
		##### Preprocessing Frame #######
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		size = frame.shape
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
		        #cv2.imshow('masked',maskedskinycb)
		        #cv2.imshow("YCrCb",skinRegionycb)
		        x1=int(landmarks[0][0])
		        for (x, y) in landmarks:
		            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			    #cv2.circle(frame, (400,200),10, (255,0,0),-1)
		  
		        newframe=frame[:,0:x1]
		        maskedframe=maskedskinycb[:,0:x1]
		        handregion=skinRegionycb[:,0:x1]
		        drawing,xh,yh,wh,hh=handconvex(handregion,newframe)
		        #print("[INFO] Extracting the Hand region...")
		        #cv2.imshow('output',drawing)
		        #crphand=newframe[yh:yh+hh,xh:xh+wh]
		        crphand=maskedframe[yh:yh+hh,xh:xh+wh]
		        #cv2.imshow('crophand',crphand)
	

			## Resizeing the image ##
		        r = 90.0 / crphand.shape[1]
		        dim = (90, int(crphand.shape[0] * r))
		        # perform the actual resizing of the image and show it
			#resized = cv2.resize(crphand, dim, interpolation = cv2.INTER_AREA)
		        resized = crphand
		        opdir='/finger/op_imgs/'
		        img_files=os.listdir(opdir)
		        len(img_files)
		        if len(img_files)<6:
                                if fr==10:
                                        cv2.imwrite(opdir+"hand-" + str(im) + ".jpg", resized)
                                        fr=1
                                elif len(img_files)==5:
                                        print("[INFO] Classifying...")
                                        prs=[]
			    ## classification
                                        for img_file in img_files:
                                            text=classifyhand(opdir+img_file)
                                            prs.append(text)
                                            os.remove(opdir+img_file)
                                            prs=sorted(prs,key=prs.count,reverse=True)
			                #print(str(prs[0]))
                                        r=requests.post("http://IP:Port", data ={'direction': str(prs[0])})
                                        print('sent'+ ':  ' + str(prs[0]))

                                        break
		        fr=fr+1
		        font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.imshow("Output", frame)
	im=im+1
	key = cv2.waitKey(1) & 0xFF
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

