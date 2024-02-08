#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import rerun as rr
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = True
# Better handling for occlusions:
lr_check = True

#add windows for debugging
DEBUG = True
 
heightFromFloor = 0.5 #meters

focalLength = 1.3 * 1e-3 #1.3 mm
lensLength = (3*1e-9 * 640, 3*1e-9 * 480)
horizontalPixelLoc = (np.indices((480,640))[1] - 320 * np.ones((480,640))) * 3e-6 
horizontalTheta = np.arctan(horizontalPixelLoc / focalLength)
verticalPixelLoc = -1 * (np.indices((480,640))[0] - 240 * np.ones((480,640))) * 3e-6 

verticalTheta =  np.arctan(verticalPixelLoc/np.sqrt(horizontalPixelLoc * horizontalPixelLoc + focalLength * focalLength))

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)


xoutDepth.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setCamera("right")

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
depth.setConfidenceThreshold(200)

config = depth.initialConfig.get()

config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 20
config.postProcessing.temporalFilter.enable = True
config.postProcessing.temporalFilter.alpha = 1

config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
depth.initialConfig.set(config)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(xoutDepth.input)

rr.init("rerun_example_depth_image", spawn=True)
rr.log(
    "world/camera",
    rr.Pinhole(
        width=640,
        height=480,
        focal_length=433,
    ),
)

finalOutput = []

bufferSize = 5
bufferIndex = 0
imageBufferArray = [None] * (bufferSize)

def addImageToBuffer(newImage):
    global bufferIndex, bufferSize, imageBufferArray
    imageBufferArray[bufferIndex] = newImage.copy()
    bufferIndex = bufferIndex + 1
    if bufferIndex >= bufferSize - 1:
        bufferIndex = 0

def getGuaranteedDepth():
    global bufferIndex, imageBufferArray
    finalImage = imageBufferArray[0].copy()
    # return imageBufferArray[bufferIndex]
    for i in range(bufferSize- 1):
        if isinstance(imageBufferArray[1+i], np.ndarray):
            finalImage = finalImage * imageBufferArray[1+i]
    return finalImage

birdsEyeViewMap = []

def getBirdsEyeViewMap():
    return birdsEyeViewMap

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the disparity frames from the outputs defined above

    q2 = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    while True:
        inDepth = q2.get()
        depthFrame = inDepth.getFrame() #gets numpy array of depths per each pixel

        depthCalc = depthFrame /1000  #converts depth from mm to meters
        
        relativeDistFromCamera = depthCalc #how far away from camera
        relativePerpendicularFromCamera = depthCalc * np.tan(horizontalTheta) #how much to the side
        distance = np.sqrt((relativeDistFromCamera * relativeDistFromCamera) + (relativePerpendicularFromCamera * relativePerpendicularFromCamera)) #distance from camera focal point
        depthCalc[depthCalc > 4] = gaussian_filter(depthCalc[depthCalc > 4], sigma=3)


        scale = 35 #how much to scale on birds eye view image
        newMap = np.zeros((480 * 2,640 * 2)) 
        newX = relativeDistFromCamera 
        newY =  (relativePerpendicularFromCamera ) + 480/scale #+480/scale puts on the left side center
        
        
        height =  distance * np.tan(verticalTheta) #height relative to camera (offset to the y axis)

        newX[height > 1- heightFromFloor ] = 0 #removes floor and ceiling from birdseye view
        newY[height > 1 - heightFromFloor ] = 0 
        newX[height < -heightFromFloor ] = 0 
        newY[height < -heightFromFloor ] = 0 
        
        inbetweenDetector = np.unique(relativeDistFromCamera)
        
        
        newX[depthCalc == 0] = 0 
        newY[depthCalc == 0] = 0
        
        newX = np.round(newX * scale).flatten().astype(int) #converts 2d array of newX cordinates to a 1D array, used to tranverse the newMap array one
        newY = np.round(newY * scale).flatten().astype(int) #^^^
        
        
        newMap[newY, newX] = 1 #if something was detected, set this pixel to white
        
        #gets rid of one pixel detections, expands slightly for donut like shapes to be filled in
        # newMap = ndimage.binary_erosion(newMap, iterations=2).astype(newMap.dtype) 
        # newMap = ndimage.binary_dilation(newMap, iterations=2).astype(newMap.dtype)
        
        addImageToBuffer(newMap)
        finalMap = getGuaranteedDepth()
        finalMap = ndimage.binary_dilation(finalMap, iterations=1).astype(finalMap.dtype)

        if DEBUG:
            cv2.imshow("field layout", finalMap)
            cv2.imshow("instant field layout", newMap)
            

            cv2.imshow("depth", depthCalc)
        
            cv2.imshow("height", np.abs(height))

            rr.log("world/camera/depth", rr.DepthImage(depthCalc, meter=10_000.0))

            cv2.imshow("height onlyvisible", depthCalc)
            
        birdsEyeViewMap = finalMap
        if cv2.waitKey(1) == ord('q'):
            break