import cv2
import numpy as np
from scipy import ndimage 
from cscore import CameraServer
import depthai as dai
import robotDetection as rd
import logging 
import datetime
from pathlib import Path
import ntcore
import AStar

currentTime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
logger = logging.getLogger()
try:
    logging.basicConfig(filename="/mnt/usblog/logs/" + currentTime + ".log", 
                format='%(asctime)s: %(levelname)s: %(message)s', 
                level=logging.DEBUG) 
except:
    logging.basicConfig(filename="/home/hoot/Robotics-AStarPathFinding/logs/" + currentTime + ".log", 
                    format='%(asctime)s: %(levelname)s: %(message)s', 
                    level=logging.DEBUG) 
    logger.info("USB not detected, saving locally")

logger.info("DepthAI Script Running (Logging started properly)")

# IP_ADDRESS = "10.12.77.67"
# IP_ADDRESS = "localhost"
# logger.info("Launcing Server at " + IP_ADDRESS)
logger.info("Connecting to network table")
inst = ntcore.NetworkTableInstance.getDefault()
depthTable = inst.getTable("depthcamera")
robotTable = inst.getTable("SmartDashboard")

inst.startClient4("depth client")
inst.setServerTeam(1277)
logger.info("Connected")
connectedBool = depthTable.getBooleanTopic("Depth Raspberry PI Connected").publish()
connectedBool.set(True)
logger.info("Updated connected status")

logger.info("Enabling Camera Stream logging")
CameraServer.enableLogging()
logger.info("Enabled")

output = CameraServer.putVideo("Depth Camera", 640, 480)


# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# logger.info("Socket created")
# s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# logger.info("Socket reuse authorized")
# s.settimeout(5)
# s.bind((IP_ADDRESS, 5000))
# logger.info("Binded to server")
# s.listen()
# logger.info("Server running")

imageName = "2024-field_binary.png"
binaryMap = Path(__file__).with_name(imageName)
binaryMapFileLoc = str(binaryMap.absolute())
logger.info("Opening field map at: " + binaryMapFileLoc)
img = cv2.imread(binaryMapFileLoc, cv2.IMREAD_GRAYSCALE)

img_config = {"top-left": [
      150,
      79
    ],
    "bottom-right": [
      2961,
      1476
    ], "field-size": [
    54.27083,
    26.9375
  ],}



robotStartPosPixel = (250,250)
robotEndPosPixel = (100,650)
discreteGrid = 5
scale = 0.25

img = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)

displayImg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
displayImg[robotStartPosPixel] = (0,255,0)
displayImg[robotEndPosPixel] = (255,0,0)

#uses the following alg 
#https://mat.uab.cat/~alseda/MasterOpt/AStar-Algorithm.pdf
#https://en.wikipedia.org/wiki/A*_search_algorithm

def feetToMeters(cord):
    newTuple = []
    for i in cord:
        newTuple.append(i /3.281) 
    return tuple(newTuple)

imgSize = np.asarray(img.shape[::-1])
fieldSize = img_config["field-size"]
fieldSizeMeters = feetToMeters(fieldSize)

topLeftFieldPixels = np.asarray(img_config["top-left"]) * scale#pixels
bottomRightFieldPixels = np.asarray(img_config["bottom-right"]) * scale#pixels

fieldInPixels = (bottomRightFieldPixels[0] - topLeftFieldPixels[0],bottomRightFieldPixels[1] - topLeftFieldPixels[1])

bottomLeftFieldMeters = ((fieldSizeMeters[0] * topLeftFieldPixels[0]/fieldInPixels[0]), (fieldSizeMeters[1] * bottomRightFieldPixels[1]/fieldInPixels[1]))

metersToPixelScale = (fieldInPixels[0]/fieldSizeMeters[0],fieldInPixels[1]/fieldSizeMeters[1])
imgSizeMeters = imgSize / metersToPixelScale
def convertRobotPoseToPixel(pos):
    print(bottomLeftFieldMeters)
    #use top left and right, as well as image size, to determine locations
    pos = (pos[0] + bottomLeftFieldMeters[0],  imgSizeMeters[1] + (fieldSizeMeters[1] - (pos[1] +  bottomLeftFieldMeters[1])))
    pixelPosition = ( int(np.round(fieldInPixels[0] *  pos[0]/fieldSizeMeters[0])), int(np.round(fieldInPixels[1] * pos[1]/fieldSizeMeters[1])))
    return pixelPosition

def convertPixelToRobotPose(pos):
    logger.info(pos)
    robotPosOffset = pos *  np.array([fieldSizeMeters[0]/(fieldInPixels[0] ), fieldSizeMeters[1]/(fieldInPixels[1])])
    robotPosOffset = robotPosOffset *  np.array([1,-1])
    robotPos = robotPosOffset +  np.array([-bottomLeftFieldMeters[0], imgSizeMeters[1] + fieldSizeMeters[1] - bottomLeftFieldMeters[1]])
    return robotPos

#https://stackoverflow.com/questions/24318078/is-there-a-faster-way-to-rotate-a-large-rgba-image-than-scipy-interpolate-rotate
def rotate_CV(image, angel , interpolation):

    '''
        input :
        image           :  image                    : ndarray
        angel           :  rotation angel           : int
        interpolation   :  interpolation mode       : cv2 Interpolation object
        
                                                        Interpolation modes :
                                                        interpolation cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR
                                                        https://theailearner.com/2018/11/15/image-interpolation-using-opencv-python/
                                                        
        returns : 
        rotated image   : ndarray
        
        '''



    #in OpenCV we need to form the tranformation matrix and apply affine calculations
    #
    h,w = image.shape[:2]
    cX,cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angel,1)
    rotated = cv2.warpAffine(image,M , (w,h),flags=interpolation)
    return rotated


def runCamera(sharedCamera):

    sharedCamera.startDepthCamera(depthCameraConfig)
    sharedCamera.runCamera(depthCameraProcessing,depthCameraConfig)
    
row, col = np.indices(img.shape)

isDepthCameraConnected = False

if __name__ == "__main__":
    logger.info("Starting depth camera")
    try:
        depthCameraConfig = rd.DepthCameraConfig()
        depthCamera = rd.DepthCamera(depthCameraConfig)
        depthCameraProcessing = rd.DepthCameraProcessing()
        depthCamera.startDepthCamera(depthCameraConfig)
        isDepthCameraConnected = True
    except:
        isDepthCameraConnected = False
    
    
 
    # sharedCamera = depthCamera('c', rd.DepthCamera)
    
    robotStartPosPixel = convertRobotPoseToPixel((1,2))
    robotEndPosPixel = convertRobotPoseToPixel((15,7))
    kernel = np.ones((25, 25), np.uint8) #approx half the robot 
    kernel2 = np.ones((3,3), np.uint8)
    testImage = displayImg.copy()
    testImage = cv2.dilate(testImage, kernel, iterations= 2)
    img = cv2.dilate(img, kernel, iterations=1)
    
    # logger.info("Waiting for roborio connection")
    # clientsocket, address = s.accept()
    # logger.info("Connections established at " + str(address))  
    # time.sleep(1)

    logger.info("Starting depth pipeline")
    if isDepthCameraConnected:
        try:
            device = dai.Device(depthCamera.pipeline)
        except:
            isDepthCameraConnected = False
    
    if isDepthCameraConnected:
        logger.info("Depth Camera Detected and running smoothly!")
    else:
        logger.info("No depth camera detected, rerun script (limited functionality for AStar without depth input)")

    depthCameraConnected = depthTable.getBooleanTopic("depth-camera-connected").publish()
    depthCameraConnected.set(isDepthCameraConnected)
    # Output queue will be used to get the disparity frames from the outputs defined above

    if isDepthCameraConnected == True:
        q2 = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    passiveMode = depthTable.getBooleanTopic("passive").subscribe(False)
    passiveModeOn = False
    passiveTopic = depthTable.getRawTopic("passive-data").publish("byte array")
    
    runCommandSub = depthTable.getBooleanTopic("Trajectory Request").subscribe(False)
    runCommandSet = depthTable.getBooleanTopic("Trajectory Request").publish()
    runEndPosSub = depthTable.getDoubleArrayTopic("Trajectory End Point").subscribe([0,0,0])
    
    informFinishedCommandSet = depthTable.getBooleanTopic("Trajectory Request Fulfilled").publish()
    
    resultStream = depthTable.getDoubleArrayTopic("Trajectory Data").publish()
    robotPositionSub = robotTable.getDoubleArrayTopic("Position").subscribe([0.0,0.0,0.0])
    # commandStream

    updateIteration = 0

    logger.info("Starting camera stream")
    while True:
        if isDepthCameraConnected == True:
            inDepth = q2.get()
            depthFrame = inDepth.getFrame() #gets numpy array of depths per each pixel
            logger.info("Line 229, processing frame")
            newMap = depthCameraProcessing.processDepthFrame(depthFrame, depthCameraConfig)
            logger.info("Line 231, got processed frame, adding to buffer")

            depthCameraProcessing.addToBuffer(newMap)
            logger.info("Line 234, Added to buffer, getting frame collective")
            finalMap = depthCameraProcessing.getGuaranteedDepth()
            # finalMap = ndimage.binary_dilation(finalMap, iterations=1).astype(finalMap.dtype)

            updateIteration +=1
            if updateIteration > 10:
                updateIteration = 0
                passiveModeOn = passiveMode.get()
                
            logger.info("Ran Loop Iteration")
            if passiveModeOn== True:
                # imgFinalMap = cv2.cvtColor(finalMap.astype('float32'),cv2.COLOR_GRAY2BGR)
                finalMap *= 255
                output.putFrame(finalMap )
                logger.info("sent image to camera server")

        # socket_info = clientsocket.recv(1024).decode("utf-8")
        # command = socket_info.strip().replace("\x00\x0f","").split(" ")
        # print(command)

        #format: "RUN ROBOTPOSX ROBOTPOSY HEADING"
        if runCommandSub.get() == True:
            runCommandSet.set(False)
            informFinishedCommandSet.set(False)
            logger.info("Fetching depth camera data")
            robotPos = robotPositionSub.get() #posx, posy, angle
            endPos = runEndPosSub.get()
            logger.info("Current robot position: " + str(robotPos))

            angle = robotPos[2]
            original_angle = 180
            robotStartPosPixel = convertRobotPoseToPixel((robotPos[0],robotPos[1]))
            robotEndPosPixel = convertRobotPoseToPixel((endPos[0], endPos[1]))

            if isDepthCameraConnected:
                finalMap = rotate_CV(finalMap, angle, cv2.INTER_LINEAR)
                maxSize = 640
                camX = np.floor(np.cos((original_angle+angle) * np.pi/180) * maxSize/2) + finalMap.shape[1]//2
                camY = -np.floor(np.sin((original_angle+angle) * np.pi/180) * maxSize/2) + finalMap.shape[0]//2
                camX = int(camX)
                camY = int(camY)
                # cv2.circle(finalMap, (camX,camY), 5, 255)
                logger.info("Got cam pos")
            
                rowShifted = row - robotStartPosPixel[1] + camY
                colShifted = col - robotStartPosPixel[0] + camX #robotStartPosPixel[1] 
                
                rowShifted = rowShifted.astype('int32')
                colShifted = colShifted.astype('int32')
                
                invalid_indices = (rowShifted < 0) | (rowShifted >= finalMap.shape[0]) | \
                                (colShifted < 0) | (colShifted >= finalMap.shape[1])
                
                rowShifted[invalid_indices] = 0
                colShifted[invalid_indices] = 0
                
                finalMap[0,0] = 0
                finalMap = cv2.dilate(finalMap, kernel2, iterations=1).astype(finalMap.dtype)
                logger.info("Got final map")
            
            if isDepthCameraConnected:
                newImg = img + finalMap[rowShifted,colShifted]
            else:
                newImg = img
                
            newImg = np.where(newImg > 0.5, 255, 0).astype(np.uint8)

            # coloredImg = cv2.cvtColor(newImg.astype('float32'),cv2.COLOR_GRAY2BGR)
            
            logger.info("Starting ASTAR")
            logger.info("Robot start pos: " + str(robotStartPosPixel))
            logger.info("Robot end pos: " + str(robotEndPosPixel))

            output.putFrame(newImg)
            result = AStar.AStar(robotStartPosPixel,robotEndPosPixel,newImg)
            # for i in range(len(result) -1):
            #     # displayImg[convertToImageCordinate(result[i])] = (255,0,255)
            #     line = cv2.line(coloredImg, result[i], result[i+1], (255,0,255), 2)
            
            
            logger.info("Sending results")
            logger.debug(result)
            finalResult = []
            if result != "FAILED":
                result = np.array(result).astype(np.float64)
                result = np.flip(result,0) #because AStar reconstruction starts from the end to the start, flip array to be start to end
                result = convertPixelToRobotPose(result)
                finalResult = list(result.ravel())
            
            resultStream.set(finalResult)
            informFinishedCommandSet.set(True)
            logger.info("Finished run")
# sys.stdout.close()
        
    
    
