#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import rerun as rr

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

focalLength = 1.3 * 1e-3 #1.3 mm
lensLength = (3*1e-9 * 640, 3*1e-9 * 480)
horizontalPixelLoc = (np.indices((480,640))[1] - 320 * np.ones((480,640))) * 3e-6 
horizontalTheta = np.arctan(horizontalPixelLoc / focalLength)
verticalPixelLoc = (np.indices((480,640))[0] - 240 * np.ones((480,640))) * 3e-6 

verticalTheta =  np.arctan(verticalPixelLoc/focalLength)
# print(np.max(np.tan(horizontalTheta)))

# depthCalc = np.zeros((480,640))

# depthCalc[200:280, 200:440] = 200
# depthCalc[220:260, 300:340] = 50


# projectionDistance = np.cos(verticalTheta) * depthCalc
# relativeDistFromCamera = projectionDistance * np.cos(horizontalTheta)
# relativePerpendicularFromCamera = projectionDistance * np.sin(horizontalTheta) #cms
# cv2.imshow("a", depthCalc)


# scale = 2
# newMap = np.zeros((480,640))
# newX = relativeDistFromCamera * scale
# newY = 480/2 + relativePerpendicularFromCamera * scale
# newX = np.round(newX).flatten().astype(int)
# newY = np.round(newY).flatten().astype(int)

# # print(newX)
# # testArray = newX

# # print(newX)
# # # print([x if x in newX if x != 0])
# # print(newY)

# # cordMap = np.vstack([(newY.T, newX.T)]).T
# # cordMap = [newY, newX]
# cordMap = np.concatenate([newX[:,None],newY[:,None]], axis=1).astype(int)
# print(cordMap)
# # cordMap = cordMap[np.unique(cordMap).astype(int)]

# # print(cordMap[0,0])
# # cordMap = cordMap.flatten()
# # cordMapMod = np.array(cordMap) 
# # print(np.unique(cordMap, axis=0))
# # print(tuple(np.vsplit(cordMapMod.T, 1)[0]))

# # cordMap = cordMap[cordMap != (240,0)]
# # print(cordMap[np.unique(cordMap, axis=0).astype(int)])
# # cordMap = cordMap.reshape(-1,1)
# # cordMap = np.array(map(tuple, cordMap))
# # cordMap = list(map(tuple, cordMap))
# # print(cordMap[np.unique(cordMap)])
# # tupleCordMap = list(map(tuple, cordMap))
# # print(tupleCordMap)
# # print(cordMap[np.unique(cordMap)])

# # idx, cnt = np.unique(cordMap, return_counts=True)
# # print(idx)
# # print(cordMap)
# print(cordMap)
# newMap[newY, newX] = 1
# # np.put(newMap, cordMap, 1)



# newMap[480/2,0] 
# newMap[np.arange(newMap.shape[0])[:, None], [newX, newY]]
# newMap[(newY, newX)] += 100

# cv2.imshow("image", newMap)
# cv2.waitKey(0)

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName("disparity")

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
depth.setSubpixel(True)
depth.setConfidenceThreshold(150)

config = depth.initialConfig.get()

config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 25
config.postProcessing.temporalFilter.enable = False
# config.postProcessing.temporalFilter.alpha = 1

config.postProcessing.spatialFilter.enable = False
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
depth.initialConfig.set(config)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)


rr.init("rerun_example_depth_image", spawn=True)
rr.log(
    "world/camera",
    rr.Pinhole(
        width=640,
        height=480,
        focal_length=433,
    ),
)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    while True:
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()
        # Normalization for better visualization
        # frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        depthCalc = (433.333333 * 7.5   / frame ).astype(np.uint8) #
        
        depthCalc = depthCalc 
        # depthCalc = cv2.fastNlMeansDenoisingColored(depthCalc,None,10,10,7,21)
        
        projectionDistance = depthCalc
        relativeDistFromCamera = depthCalc
        relativePerpendicularFromCamera = depthCalc * np.tan(horizontalTheta) #cms


        scale = 3
        newMap = np.zeros((480 * 2,640 * 2))
        newX = relativeDistFromCamera * scale
        newY =  (relativePerpendicularFromCamera  * scale) + 480
        
        
        distance = np.sqrt(relativeDistFromCamera * relativeDistFromCamera + relativePerpendicularFromCamera * relativePerpendicularFromCamera)
        height = np.abs(distance * np.tan(verticalTheta))

        # newX[height > 0.15 ] = 0 #3 feet = ~95 cm
        # newY[height > 0.15 ] = 0 
        
        newX[depthCalc == 0] = 0
        newY[depthCalc == 0] = 0
        
        
        newX = np.round(newX).flatten().astype(int)
        newY = np.round(newY).flatten().astype(int)
        print(np.unique(newX))

        newMap[newY, newX] = 1
        
        cv2.imshow("field layout", newMap)


        cv2.imshow("disparity", depthCalc)
        
        cv2.imshow("distance", relativeDistFromCamera)
        cv2.imshow("perpendicular", np.abs(relativePerpendicularFromCamera)/np.max(relativePerpendicularFromCamera))
        


        cv2.imshow("height", height/np.max(height))


        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        # cv2.imshow("disparity_color", frame)

        # cv2.imshow("depth", depthCalc)
        
        rr.log("world/camera/depth", rr.DepthImage(depthCalc, meter=10_000.0))

        # depthCalc
        
        h = depthCalc.shape[0]
        w = depthCalc.shape[1]
        
        # ThetaXGraph= numpy.arrange(-319,320, 1)
        # loop over the image, pixel by pixel
        # for y in range(0, h):
        #     for x in range(0, w):
        #         # threshold the pixel
        #         PLx = np.abs(lensLength[0] / 2 - x)
        #         PLy = np.abs(lensLength[1]/2 - y)
        #         ThetaX = np.arctan2(PLx,focalLength)
        #         ThetaY = np.arctan2(PLy,np.sqrt(PLx**2 + focalLength**2))
        #         projV = np.cos(ThetaY) * depthCalc[y,x]
        #         relativeDistFromCamera = projV * np.cos(ThetaX)
        #         relativePerpendicularFromCamera = projV * np.sin(ThetaX) #cms
                
        #         newX = relativeDistFromCamera /2
        #         newY = h/2 + relativePerpendicularFromCamera /2
        #         newX = int(np.round(newX))
        #         newY = int(np.round(newY))

        #         # newMap[h/2,0] 
        #         newMap[newY, newX] += 100
            

        
        if cv2.waitKey(1) == ord('q'):
            break