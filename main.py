import cv2
import numpy as np
from scipy import ndimage 
from cscore import CameraServer
import depthai as dai
import time
import robotDetection as rd
import logging 
import datetime
from pathlib import Path
import ntcore

currentTime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
logger = logging.getLogger()
try:
    logging.basicConfig(filename="/mnt/usblog/logs/" + currentTime + ".log", 
                format='%(asctime)s: %(levelname)s: %(message)s', 
                level=logging.DEBUG) 
except:
    logging.basicConfig(filename="logs/" + currentTime + ".log", 
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
connectedBool = depthTable.getBooleanTopic("Depth Camera Connected").publish()
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

def dist(start, end):
    return np.power((start[0] - end[0])**2 + (start[1]-end[1])**2, 0.5)
    

def reconstructPath(came_from, current_node):
    print("HERE")
    total_path = [current_node]
    while current_node in came_from:
        current_node = came_from[current_node]
        total_path.append(current_node)
    return total_path
    
def getNeighboringTuples(node, binaryImage):
    neighboringTuples = []
    for i in [-discreteGrid,0,discreteGrid]:
        for j in [-discreteGrid,0,discreteGrid]:
            if i != 0 or j != 0:
                new_node = (node[0] + i, node[1] + j)
                if binaryImage[convertToImageCordinate(new_node)] == 0:
                    neighboringTuples.append(new_node)
    return neighboringTuples

def convertToImageCordinate(cord):
    return (cord[1], cord[0])

def AStar(start, end, binaryImg):
    refreshIndex = 0
    refreshRate = 10 #HZ
    binaryImage = binaryImg
    open_nodes = [start]
    came_from = {}
    gScore = {}
    gScore[start] = 0
    fScore = {}
    fScore[start] = dist(start, end)
    
    while open_nodes != []:
        current_node = min(open_nodes, key = lambda y: fScore.get(y,1e99))
        if dist(current_node, end) < discreteGrid:
            return reconstructPath(came_from, current_node)
        open_nodes.remove(current_node)
        neighboring_nodes = getNeighboringTuples(current_node, binaryImage)
        for neighbor in neighboring_nodes:
            tentative_gScore = gScore.get(current_node, 1e99) + dist(current_node, neighbor)
            if tentative_gScore < gScore.get(neighbor,1e99):
                came_from[neighbor] = current_node
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + dist(neighbor,end)
                if neighbor not in open_nodes:
                    open_nodes.append(neighbor)


    return "FAILED"

def runAStar():
    print(AStar(robotStartPosPixel,robotEndPosPixel))


fieldSize = img_config["field-size"]
fieldSizeMeters = feetToMeters(fieldSize)

topLeftFieldPixels = img_config["top-left"] #pixels
bottomRightFieldPixels = img_config["bottom-right"] #pixels

fieldInPixels = (bottomRightFieldPixels[0] - topLeftFieldPixels[0],bottomRightFieldPixels[1] - topLeftFieldPixels[1])

bottomLeftFieldMeters = ((fieldSizeMeters[0] * topLeftFieldPixels[0]/fieldInPixels[0]), (fieldSizeMeters[1] * bottomRightFieldPixels[1]/fieldInPixels[1]))

metersToPixelScale = (fieldInPixels[0]/fieldSizeMeters[0],fieldInPixels[1]/fieldSizeMeters[1])

def convertRobotPoseToPixel(pos):
    print(bottomLeftFieldMeters)
    #use top left and right, as well as image size, to determine locations
    pos = (pos[0] + bottomLeftFieldMeters[0],  fieldSizeMeters[1] + (fieldSizeMeters[1] - (pos[1] +  bottomLeftFieldMeters[1])))
    pixelPosition = ( int(np.round(scale * fieldInPixels[0] *  pos[0]/fieldSizeMeters[0])), int(np.round(scale * fieldInPixels[1] * pos[1]/fieldSizeMeters[1])))
    return pixelPosition

def convertPixelToRobotPose(pos):
    robotPosOffset = (pos[0] * fieldSizeMeters[0]/(fieldInPixels[0] * scale), pos[1] * fieldSizeMeters[1]/(fieldInPixels[1] * scale))
    robotPosOffset = (pos[0] * fieldSizeMeters[0]/(fieldInPixels[0] * scale), pos[1] * fieldSizeMeters[1]/(fieldInPixels[1] * scale))
    

def runCamera(sharedCamera):

    sharedCamera.startDepthCamera(depthCameraConfig)
    sharedCamera.runCamera(depthCameraProcessing,depthCameraConfig)
    

if __name__ == "__main__":
    logger.info("Starting depth camera")
    depthCameraConfig = rd.DepthCameraConfig()
    depthCamera = rd.DepthCamera(depthCameraConfig)
    depthCameraProcessing = rd.DepthCameraProcessing()
    depthCamera.startDepthCamera(depthCameraConfig)
    
 
    # sharedCamera = depthCamera('c', rd.DepthCamera)
    
    robotStartPosPixel = convertRobotPoseToPixel((1,2))
    robotEndPosPixel = convertRobotPoseToPixel((15,7))
    kernel = np.ones((15, 15), np.uint8) 
    testImage = displayImg.copy()
    testImage = cv2.dilate(testImage, kernel, iterations= 1)
    
    # logger.info("Waiting for roborio connection")
    # clientsocket, address = s.accept()
    # logger.info("Connections established at " + str(address))  
    # time.sleep(1)

    logger.info("Starting pipeline")
    with dai.Device(depthCamera.pipeline) as device:

        # Output queue will be used to get the disparity frames from the outputs defined above

        q2 = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        passiveMode = depthTable.getBooleanTopic("passive").subscribe(False)
        passiveTopic = depthTable.getRawTopic("passive-data").publish("byte array")
        
        runCommandSub = depthTable.getBooleanTopic("Calculate Trajectory").subscribe(False)
        runCommandSet = depthTable.getBooleanTopic("Calculate Trajectory").publish()
        
        resultStream = depthTable.getRawTopic("Trajectory").publish("byte array")
        # commandStream

        logger.info("Starting camera stream")
        while True:
            inDepth = q2.get()
            depthFrame = inDepth.getFrame() #gets numpy array of depths per each pixel

            newMap = depthCameraProcessing.processDepthFrame(depthFrame, depthCameraConfig)
              
            depthCameraProcessing.addToBuffer(newMap)
            finalMap = depthCameraProcessing.getGuaranteedDepth()
            # finalMap = ndimage.binary_dilation(finalMap, iterations=1).astype(finalMap.dtype)

            logger.info("Ran Loop Iteration")
            if passiveMode.get() == True:
                # imgFinalMap = cv2.cvtColor(finalMap.astype('float32'),cv2.COLOR_GRAY2BGR)
                finalMap *= 255
                logger.info(np.unique(finalMap))
                output.putFrame(finalMap )
            # socket_info = clientsocket.recv(1024).decode("utf-8")
            # command = socket_info.strip().replace("\x00\x0f","").split(" ")
            # print(command)

            #format: "RUN ROBOTPOSX ROBOTPOSY HEADING"
            if runCommandSub.get() == True:
                runCommandSet.set(False)
                logger.info("Fetching depth camera data")
                startTime = datetime.now() #used to see time it takes to run
                robotPos = robotTable.getEntry("Position") #posx, posy, angle
                angle = robotPos[2]
                original_angle = 180
                finalMap = ndimage.rotate(finalMap, angle)
                print(finalMap.shape)
                maxSize = 640
                camX = np.floor(np.cos((original_angle+angle) * np.pi/180) * maxSize/2) + finalMap.shape[1]//2
                camY = -np.floor(np.sin((original_angle+angle) * np.pi/180) * maxSize/2) + finalMap.shape[0]//2
                camX = int(camX)
                camY = int(camY)
                print((camX,camY))
                cv2.circle(finalMap, (camX,camY), 5, 255)
                
                row, col = np.indices(img.shape)
                
                robotStartPosPixel = (robotPos[0],robotPos[1])
                print(robotStartPosPixel)
                rowShifted = row - robotStartPosPixel[1] + camY
                colShifted = col - robotStartPosPixel[0] + camX #robotStartPosPixel[1] 
                rowShifted = rowShifted.astype('int32')
                colShifted = colShifted.astype('int32')
                print(rowShifted)
                rowShifted[rowShifted >= finalMap.shape[0]] = 0
                colShifted[colShifted >= finalMap.shape[1]] = 0
                rowShifted[rowShifted <= 0] = 0
                colShifted[colShifted <= 0] = 0
                rowShifted[colShifted <= 0] = 0
                colShifted[rowShifted <= 0] = 0
                finalMap[0,0] = 0
                
                newImg = img + finalMap[rowShifted,colShifted]
                newImg[newImg > 0.5] = 255
                newImg[newImg < 0.5] = 0

                # coloredImg = cv2.cvtColor(newImg.astype('float32'),cv2.COLOR_GRAY2BGR)
                newImg = ndimage.binary_dilation(newImg, iterations=2).astype(finalMap.dtype)
                result = AStar(robotStartPosPixel,robotEndPosPixel,newImg)
                # for i in range(len(result) -1):
                #     # displayImg[convertToImageCordinate(result[i])] = (255,0,255)
                #     line = cv2.line(coloredImg, result[i], result[i+1], (255,0,255), 2)
                
                logger.info("Sending results")
                logger.debug(result)
                resultStream.setRaw("Trajectory", str(result), "utf-8")

                endTime = datetime.now() #used to see time it takes to run
                logger.info("Finished run, took ", endTime-startTime)
    # sys.stdout.close()
            
        
        
