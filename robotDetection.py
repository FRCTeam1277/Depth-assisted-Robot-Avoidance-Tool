#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import logging
logger = logging.getLogger()

class DepthCamera:
    def __init__(self, depthCameraConfig):
        self._birdsEyeViewMap = []
    
    def startDepthCamera(self, depthCameraConfig): 
        """Creates pipeline for camera

        Args:
            depthCameraConfig (_type_): _description_
        """
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
        self.metersToPixelScaleX = 170//4 #Todo Figure out how these values were obtained
        self.metersToPixelScaleY = 170//4
        self.height_from_floor_meters = 0.5 #meter cut off, also used for the ceiling
        
        self.camera_width_pixels = 640
        self.camera_height_pixels = 480

class DepthCameraProcessing:
    def __init__(self, depthCameraConfig:DepthCameraConfig):
        """Class that will compute birds eye view
        """
        ##VV all are constants or prebaked constants, so during run time we do minimal computations
        ##This saves A LOT of time!! Made the camera go from like 4 fps to 16ish
        self.focalLength = 1.3 * 1e-3 #1.3 mm
        self.lensLength = (3*1e-9 * depthCameraConfig.camera_width_pixels, 3*1e-9 * depthCameraConfig.camera_height_pixels)
        self.horizontalPixelLoc = (np.indices((depthCameraConfig.camera_height_pixels,depthCameraConfig.camera_width_pixels))[1] - depthCameraConfig.camera_width_pixels/2 * np.ones((depthCameraConfig.camera_height_pixels,depthCameraConfig.camera_width_pixels))) * 3e-6 
        self.horizontalTheta = np.arctan(self.horizontalPixelLoc / self.focalLength)
        self.tanHorizontalTheta = np.tan(self.horizontalTheta) #ok ... don't judge, don't want to risk breaking anything
        self.tanHorizontalThetaSquared = self.tanHorizontalTheta * self.tanHorizontalTheta 
        self.normalDistance = np.sqrt(1+ self.tanHorizontalThetaSquared)
        self.verticalPixelLoc = -1 * (np.indices((depthCameraConfig.camera_height_pixels,depthCameraConfig.camera_width_pixels))[0] - depthCameraConfig.camera_height_pixels/2 * np.ones((depthCameraConfig.camera_height_pixels,depthCameraConfig.camera_width_pixels))) * 3e-6 
        self.verticalTheta =  np.arctan(self.verticalPixelLoc/np.sqrt(self.horizontalPixelLoc * self.horizontalPixelLoc + self.focalLength * self.focalLength))
        self.tanVerticalTheta = np.tan(self.verticalTheta)
        
        self.bufferSize = 5
        self._bufferIndex = 0
        self.imageBufferArray = [None] * (self.bufferSize)

         
    def processDepthFrame(self, depthFrame, depthCameraConfig:DepthCameraConfig):
        """Creates an initial birds eye view map based off of a depth frame

        Args:
            depthFrame ([int,int, float]): Depth camera frame (POV is the camera)
            depthCameraConfig (DepthCameraConfig): config of our camera

        Returns:
            [int,int,bool]: raw depth image (better to use getGuaranteedDepth though!)
        """
        depthCalc = depthFrame /1000  #converts depth from mm to meters
        height_from_floor_meters = depthCameraConfig.height_from_floor_meters
        scaleX = depthCameraConfig.metersToPixelScaleX    #how much to scale on birds eye view image
        scaleY = depthCameraConfig.metersToPixelScaleY    #how much to scale on birds eye view image

        relativeDistFromCamera = depthCalc #how far away from camera
        relativePerpendicularFromCamera = depthCalc * self.tanHorizontalTheta #how much to the side
        distance = depthCalc * self.normalDistance  #np.sqrt((relativeDistFromCamera * relativeDistFromCamera) + (relativePerpendicularFromCamera * relativePerpendicularFromCamera)) #distance from camera focal point

        depthCalc[depthCalc > 4] = 0  #gaussian_filter(depthCalc[depthCalc > 4], sigma=3)

        newMap = np.empty((depthCameraConfig.camera_height_pixels,depthCameraConfig.camera_width_pixels)) 
        newX = relativeDistFromCamera 
        newY =  (relativePerpendicularFromCamera ) + depthCameraConfig.camera_height_pixels/scaleY #+480/scale puts on the left side center

        height =  distance * self.tanVerticalTheta #height relative to camera (offset to the y axis)

        newX[height > 1- height_from_floor_meters ] = 0 #removes floor and ceiling from birdseye view
        newY[height > 1 - height_from_floor_meters ] = 0 
        newX[height < -height_from_floor_meters ] = 0 
        newY[height < -height_from_floor_meters ] = 0 

        newX = (newX.ravel() * scaleX).astype(np.int32)
        newY = (newY.ravel() * scaleY).astype(np.int32)
        
        #newX = np.round(newX * scaleX).ravel().astype(np.int32) #converts 2d array of newX cordinates to a 1D array, used to tranverse the newMap array one
        #newY = np.round(newY * scaleY).ravel().astype(np.int32) #^^^
        
        newX[newX < 0 ] = 0
        newX[newX >= depthCameraConfig.camera_width_pixels] = 0
        newY[newY >= depthCameraConfig.camera_height_pixels] = 0
        newY[newY < 0] = 0
        
        newMap[newY, newX] = 1 #if something was detected, set this pixel to white
        
        return newMap

    def addToBuffer(self, newImage):
        """Add a raw depth image to our buffer
        """
        bufferIndex = self._bufferIndex
        self.imageBufferArray[bufferIndex] = newImage.copy()
        self.bufferIndex = bufferIndex + 1
        if bufferIndex >= self.bufferSize - 1:
            bufferIndex = 0

    def getGuaranteedDepth(self):
        """Gets a final map that is a product of all 5 images in the buffer
        This ensures that if an object is detected, it must have been detected in all 5 images before
        Other wise it is not reported.
        Makes our results more guaranteed
        """
        finalImage = self.imageBufferArray[0].copy()
        # return imageBufferArray[bufferIndex]
        for i in range(self.bufferSize- 1):
            if isinstance(self.imageBufferArray[1+i], np.ndarray):
                finalImage = finalImage * self.imageBufferArray[1+i] #if detected, we multiply by 1, if not, by 0
        return finalImage

if __name__ == "__main__":
    
    depthCameraConfig = DepthCameraConfig()

    depthCamera = DepthCamera(depthCameraConfig)
    depthCameraProcessing = DepthCameraProcessing(depthCameraConfig)

    depthCamera.startDepthCamera(depthCameraConfig)
    depthCamera.runCamera(depthCameraProcessing,depthCameraConfig)