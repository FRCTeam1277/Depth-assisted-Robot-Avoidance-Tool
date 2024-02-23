#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
# import rerun as rr
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import logging
logger = logging.getLogger()
class DepthCamera:
    def __init__(self, depthCameraConfig):
        self._birdsEyeViewMap = []
    
    def startDepthCamera(self, depthCameraConfig): 
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        depth = self.pipeline.create(dai.node.StereoDepth)
        xoutDepth = self.pipeline.create(dai.node.XLinkOut)


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

        depth.setLeftRightCheck(depthCameraConfig.lr_check)
        depth.setExtendedDisparity(depthCameraConfig.extended_disparity)
        depth.setSubpixel(depthCameraConfig.subpixel)
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
        
   
        
    def runCamera(self, depthCameraProcessing, depthCameraConfig):
        
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            # Output queue will be used to get the disparity frames from the outputs defined above

            q2 = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

            while True:
                inDepth = q2.get()
                depthFrame = inDepth.getFrame() #gets numpy array of depths per each pixel

                newMap = depthCameraProcessing.processDepthFrame(depthFrame, depthCameraConfig)
                
                #gets rid of one pixel detections, expands slightly for donut like shapes to be filled in
                # newMap = ndimage.binary_erosion(newMap, iterations=2).astype(newMap.dtype) 
                # newMap = ndimage.binary_dilation(newMap, iterations=2).astype(newMap.dtype)
                
                depthCameraProcessing.addToBuffer(newMap)
                finalMap = depthCameraProcessing.getGuaranteedDepth()
                finalMap = ndimage.binary_dilation(finalMap, iterations=1).astype(finalMap.dtype)

                if depthCameraConfig.DEBUG:
                    cv2.imshow("field layout", finalMap)
                    cv2.imshow("instant field layout", newMap)
                    

                    # cv2.imshow("depth", depthCalc)
                
                    # cv2.imshow("height", np.abs(height))

                    # rr.log("world/camera/depth", rr.DepthImage(depthCalc, meter=10_000.0))

                    # cv2.imshow("height onlyvisible", depthCalc)
                    
                self._birdsEyeViewMap = finalMap.astype(finalMap.dtype)
                if cv2.waitKey(1) == ord('q'):
                    break
    def getFinalMap(self):
        return self._birdsEyeViewMap
     
class DepthCameraConfig:
    def __init__(self):
        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        self.extended_disparity = False
        # Better accuracy for longer distance, fractional disparity 32-levels:
        self.subpixel = True
        # Better handling for occlusions:
        self.lr_check = True
        #add windows for debugging
        self.DEBUG = True
        #scale used to convert meters to a pixel array (important to set before as there is rounding error)
        self.metersToPixelScaleX = 170//4
        self.metersToPixelScaleY = 170//4
        self.heightFromFloor = 0.5 #meters

class DepthCameraProcessing:
    def __init__(self):
        self.focalLength = 1.3 * 1e-3 #1.3 mm
        self.lensLength = (3*1e-9 * 640, 3*1e-9 * 480)
        self.horizontalPixelLoc = (np.indices((480,640))[1] - 320 * np.ones((480,640))) * 3e-6 
        self.horizontalTheta = np.arctan(self.horizontalPixelLoc / self.focalLength)
        self.tanHorizontalTheta = np.tan(self.horizontalTheta) #ok ... don't judge, don't want to risk breaking anything
        self.tanHorizontalThetaSquared = self.tanHorizontalTheta * self.tanHorizontalTheta 
        self.normalDistance = np.sqrt(1+ self.tanHorizontalThetaSquared)
        self.verticalPixelLoc = -1 * (np.indices((480,640))[0] - 240 * np.ones((480,640))) * 3e-6 
        self.verticalTheta =  np.arctan(self.verticalPixelLoc/np.sqrt(self.horizontalPixelLoc * self.horizontalPixelLoc + self.focalLength * self.focalLength))
        self.tanVerticalTheta = np.tan(self.verticalTheta)
        
        self.bufferSize = 5
        self._bufferIndex = 0
        self.imageBufferArray = [None] * (self.bufferSize)

         
    def processDepthFrame(self, depthFrame, depthCameraConfig):
        depthCalc = depthFrame /1000  #converts depth from mm to meters
        heightFromFloor = depthCameraConfig.heightFromFloor
        scaleX = depthCameraConfig.metersToPixelScaleX    #how much to scale on birds eye view image
        scaleY = depthCameraConfig.metersToPixelScaleY    #how much to scale on birds eye view image

        logger.info("before distance")
        relativeDistFromCamera = depthCalc #how far away from camera
        relativePerpendicularFromCamera = depthCalc * self.tanHorizontalTheta #how much to the side
        distance = depthCalc * self.normalDistance  #np.sqrt((relativeDistFromCamera * relativeDistFromCamera) + (relativePerpendicularFromCamera * relativePerpendicularFromCamera)) #distance from camera focal point
        logger.info("after distance")

        depthCalc[depthCalc > 4] = 0  #gaussian_filter(depthCalc[depthCalc > 4], sigma=3)

        newMap = np.empty((480,640)) 
        newX = relativeDistFromCamera 
        newY =  (relativePerpendicularFromCamera ) + 240/scaleY #+480/scale puts on the left side center

        height =  distance * self.tanVerticalTheta #height relative to camera (offset to the y axis)

        newX[height > 1- heightFromFloor ] = 0 #removes floor and ceiling from birdseye view
        newY[height > 1 - heightFromFloor ] = 0 
        newX[height < -heightFromFloor ] = 0 
        newY[height < -heightFromFloor ] = 0 

        logger.info("before ruound")
        newX = (newX.ravel() * scaleX).astype(np.int32)
        newY = (newY.ravel() * scaleY).astype(np.int32)
        
        #newX = np.round(newX * scaleX).ravel().astype(np.int32) #converts 2d array of newX cordinates to a 1D array, used to tranverse the newMap array one
        #newY = np.round(newY * scaleY).ravel().astype(np.int32) #^^^
        logger.info("after ruound")
        
        newX[newX < 0 ] = 0
        newX[newX >= 640] = 0
        newY[newY >= 480] = 0
        newY[newY < 0] = 0
        
        newMap[newY, newX] = 1 #if something was detected, set this pixel to white
        
        return newMap

    def addToBuffer(self, newImage):
        bufferIndex = self._bufferIndex
        self.imageBufferArray[bufferIndex] = newImage.copy()
        self.bufferIndex = bufferIndex + 1
        if bufferIndex >= self.bufferSize - 1:
            bufferIndex = 0

    def getGuaranteedDepth(self):
        finalImage = self.imageBufferArray[0].copy()
        # return imageBufferArray[bufferIndex]
        for i in range(self.bufferSize- 1):
            if isinstance(self.imageBufferArray[1+i], np.ndarray):
                finalImage = finalImage * self.imageBufferArray[1+i]
        return finalImage

if __name__ == "__main__":
    
    depthCameraConfig = DepthCameraConfig()

    depthCamera = DepthCamera(depthCameraConfig)
    depthCameraProcessing = DepthCameraProcessing()

    depthCamera.startDepthCamera(depthCameraConfig)
    depthCamera.runCamera(depthCameraProcessing,depthCameraConfig)