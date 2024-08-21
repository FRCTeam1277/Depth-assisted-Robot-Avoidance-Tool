import cv2
import numpy as np
from scipy import ndimage 
from AStarPython import AStarOptions
from cscore import CameraServer
import depthai as dai
import robotDetection as rd
import logging 
import datetime
from pathlib import Path
import ntcore
import AStar

## Logging Tools
def startLogToUSB(logger,usbLogFile,localLogFile):
    """Saves logs to a local usb on the raspberry pi to easily debug issues
    If no usb is detected at the location specified, it will save locally on the pi
    Args:
        logger (Logger): Logger that is saving all data
        usbLogFile (str): Location of usb on raspberry pi
        localLogFile (str): Back up location on raspberry pi (local)
    """
    currentTime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    try:
        logging.basicConfig(filename=usbLogFile + currentTime + ".log", 
                    format='%(asctime)s: %(levelname)s: %(message)s', 
                    level=logging.DEBUG) 
    except:
        logging.basicConfig(filename=localLogFile + currentTime + ".log", 
                        format='%(asctime)s: %(levelname)s: %(message)s', 
                        level=logging.DEBUG) 
        logger.info("USB not detected, saving locally")
        
## Conversion Tools
def convertRobotPoseToPixel(currentPosition,bottom_left_field_meters, image_size_meters, field_size_meters, field_size_pixels):
    """Converts a robot's current position on the field (in meters) into pixel cordiantes on the bird eye view map
    Args:
        pos (_type_): current robot position on field
        bottomLeftFieldMeters (_type_): _description_
        imgSizeMeters (_type_): _description_
        fieldSizeMeters (_type_): _description_
        fieldInPixels (_type_): _description_

    Returns:
        _type_: _description_
    """
    #use top left and right, as well as image size, to determine locations
    currentPosition = (currentPosition[0] + bottom_left_field_meters[0],  image_size_meters[1] + (field_size_meters[1] - (currentPosition[1] +  bottom_left_field_meters[1])))
    pixelPosition = ( int(np.round(field_size_pixels[0] *  currentPosition[0]/field_size_meters[0])), int(np.round(field_size_pixels[1] * currentPosition[1]/field_size_meters[1])))
    return pixelPosition

def convertPixelToRobotPose(pos,bottomLeftFieldMeters, imgSizeMeters,fieldSizeMeters,fieldInPixels):
    robotPosOffset = pos *  np.array([fieldSizeMeters[0]/(fieldInPixels[0] ), fieldSizeMeters[1]/(fieldInPixels[1])])
    robotPosOffset = robotPosOffset *  np.array([1,-1])
    robotPos = robotPosOffset +  np.array([-bottomLeftFieldMeters[0], imgSizeMeters[1] + fieldSizeMeters[1] - bottomLeftFieldMeters[1]])
    return robotPos

def feetToMeters(cord):
    newTuple = []
    for i in cord:
        newTuple.append(i /3.281) 
    return tuple(newTuple)

#https://stackoverflow.com/questions/24318078/is-there-a-faster-way-to-rotate-a-large-rgba-image-than-scipy-interpolate-rotate
def rotate_CV(image, angel, interpolation):
    h,w = image.shape[:2]
    cX,cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angel,1)
    rotated = cv2.warpAffine(image,M , (w,h),flags=interpolation)
    return rotated
         
#Camera functions
def runCamera(sharedCamera, depthCameraConfig, depthCameraProcessing):
    sharedCamera.startDepthCamera(depthCameraConfig)
    sharedCamera.runCamera(depthCameraProcessing,depthCameraConfig)
    

#main function
def run(usbLogFile,localLogFile,team, image_field_config, image_scale = 0.25, display_debug = False, binaryFieldImageLocalFileLocation="2024-field_binary.png", cameraNetworkTable="depthcamera"):
    ##INTALIZATION STAGE -> SETTING UP ALL VARIABLES TO BE USED IN RUNNING STAGE
    #Setting up logger
    logger = logging.getLogger()
    startLogToUSB(logger, usbLogFile,localLogFile)
    logger.info("DepthAI Script Running (Logging started properly)")
    
    #getting network tables to communicate with robot
    logger.info("Connecting to network table")
    networktable_instance = ntcore.NetworkTableInstance.getDefault()
    table_depth = networktable_instance.getTable(cameraNetworkTable)
    table_robot = networktable_instance.getTable("SmartDashboard")
    networktable_instance.startClient4("depth client")
    networktable_instance.setServerTeam(team)
    logger.info("Connected")
    
    #tells robot we are connected
    connectedBool = table_depth.getBooleanTopic("Depth Raspberry PI Connected").publish()
    connectedBool.set(True)
    logger.info("Updated connected status")

    #enables camera debugging
    logger.info("Enabling Camera Stream logging")
    CameraServer.enableLogging()
    logger.info("Enabled")

    #provides depth camera output for debugging
    depthCameraViewOutput = CameraServer.putVideo("Depth Camera", 640, 480)
    
    #sets up A* options
    latticeLength = 5
    astar_options = AStarOptions(latticeLength)

    #loads current prebaked binary map of the field 
    binaryMap = Path(__file__).with_name(binaryFieldImageLocalFileLocation)
    binaryMapFileLoc = str(binaryMap.absolute())
    logger.info("Opening field map at: " + binaryMapFileLoc)
    binary_image = cv2.imread(binaryMapFileLoc, cv2.IMREAD_GRAYSCALE)
    binary_image = cv2.resize(binary_image,(0,0), fx=image_scale, fy=image_scale, interpolation = cv2.INTER_AREA)

    if display_debug:
        colored_image_display = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2RGB) #used for debugging

    binary_image_size_pixels = np.asarray(binary_image.shape[::-1])
    field_size_feet = image_field_config["field-size"]
    field_size_meters = feetToMeters(field_size_feet)

    #defining borders (image map is slightly bigger than the actual playing field)
    #Thus we need to add a border where the robot can't be at 
    # Also the 0,0 meter point is the bottom left corner of the field, not the 0,0 pixel location, thus we will need to offset this)
    
    #gets the top left position of the field in pixels (the last place a robot can physically be)
    top_left_field_position_pixels = np.asarray(image_field_config["top-left"]) * image_scale#pixels
    #gets the bottom right position of the field in pixels
    bottom_right_field_position_pixels = np.asarray(image_field_config["bottom-right"]) * image_scale#pixels

    #gets field length in pixels
    field_size_pixels = (bottom_right_field_position_pixels[0] - top_left_field_position_pixels[0],bottom_right_field_position_pixels[1] - top_left_field_position_pixels[1])
    
    #gets bottom left field position which is (x of top left, y of bottom right) = (left,bottom)
    bottom_left_field_position_meters = ((field_size_meters[0] * top_left_field_position_pixels[0]/field_size_pixels[0]), (field_size_meters[1] * bottom_right_field_position_pixels[1]/field_size_pixels[1]))
    
    #gets scale of how many pixels per meter (we are NOT assuming uniform scaling in both axis, but they should be simialr)
    meters_to_pixel_scale = (field_size_pixels[0]/field_size_meters[0],field_size_pixels[1]/field_size_meters[1])
    
    #gets image size in meters
    image_size_meters = binary_image_size_pixels / meters_to_pixel_scale
 
    #generates two arrays, going 0,1,2,3,4,5,. See function examples
    #This will be used later on to calculate angles between TODO!!!!
    row_array, column_array = np.indices(binary_image.shape)
    
    

    ## RUNNING CAMERA
    isDepthCameraConnected = False
    logger.info("Starting depth camera")
    try:
        depthCameraConfig = rd.DepthCameraConfig()
        depthCamera = rd.DepthCamera(depthCameraConfig)
        depthCameraProcessing = rd.DepthCameraProcessing()
        depthCamera.startDepthCamera(depthCameraConfig)
        isDepthCameraConnected = True
    except:
        isDepthCameraConnected = False
    
    
    kernel = np.ones((25, 25), np.uint8) #approx half the robot 
    kernel2 = np.ones((3,3), np.uint8)
    
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    logger.info("Starting depth camera pipeline")
    if isDepthCameraConnected:
        try:
            device = dai.Device(depthCamera.pipeline)
        except:
            isDepthCameraConnected = False
    
    if isDepthCameraConnected:
        logger.info("Depth Camera Detected and running smoothly!")
    else:
        logger.info("No depth camera detected, rerun script (limited functionality for AStar without depth input)")

    #Tells network table that we have a feed from the depth camera
    depthCameraConnected = table_depth.getBooleanTopic("depth-camera-connected").publish()
    depthCameraConnected.set(isDepthCameraConnected)
    
    # Output queue will be used to get the disparity frames from the outputs defined above
    if isDepthCameraConnected == True:
        disparityOutputQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    passiveMode = table_depth.getBooleanTopic("passive").subscribe(False)
    passiveModeOn = False
    passiveTopic = table_depth.getRawTopic("passive-data").publish("byte array")
    
    runCommandSub = table_depth.getBooleanTopic("Trajectory Request").subscribe(False)
    runCommandSet = table_depth.getBooleanTopic("Trajectory Request").publish()
    runEndPosSub = table_depth.getDoubleArrayTopic("Trajectory End Point").subscribe([0,0,0])
    
    informFinishedCommandSet = table_depth.getBooleanTopic("Trajectory Request Fulfilled").publish()
    
    resultStream = table_depth.getDoubleArrayTopic("Trajectory Data").publish()
    robotPositionSub = table_robot.getDoubleArrayTopic("Position").subscribe([0.0,0.0,0.0])

    updateIteration = 0

    #RUNS A* AND CAMERA LOOP
    logger.info("Starting camera stream")
    while True:
        if isDepthCameraConnected == True:
            inDepth = disparityOutputQueue.get()
            depthFrame = inDepth.getFrame() #gets numpy array of depths per each pixel
            logger.info("Line 229, processing frame")
            newMap = depthCameraProcessing.processDepthFrame(depthFrame, depthCameraConfig)
            logger.info("Line 231, got processed frame, adding to buffer")

            depthCameraProcessing.addToBuffer(newMap)
            logger.info("Line 234, Added to buffer, getting frame collective")
            
            #bird eye view of what is in front of the robot
            bird_eye_view_map = depthCameraProcessing.getGuaranteedDepth() 
            # finalMap = ndimage.binary_dilation(finalMap, iterations=1).astype(finalMap.dtype)

            #constantly checks if passive mode is activated by user on the driver station
            updateIteration +=1
            if updateIteration > 10:
                updateIteration = 0
                passiveModeOn = passiveMode.get()
                
            logger.info("Ran Loop Iteration")
            
            #Used for debugging, shows binary map to driver station (uses up bandwidth though!)
            if passiveModeOn== True:
                # imgFinalMap = cv2.cvtColor(finalMap.astype('float32'),cv2.COLOR_GRAY2BGR)
                bird_eye_view_map *= 255
                depthCameraViewOutput.putFrame(bird_eye_view_map)
                logger.info("sent image to camera server")

        #generates astar path when ordered to
        #to actively avoid obsticles, send run command via network table constantly!
        #format: "RUN ROBOTPOSX ROBOTPOSY HEADING"
        if runCommandSub.get() == True:
            #tells drive station that we are intalizing this function
            runCommandSet.set(False)
            informFinishedCommandSet.set(False)
            logger.info("Fetching depth camera data")
            
            #gets data from network table of where we are and where to go
            robotPos = robotPositionSub.get() #posx, posy, angle
            endPos = runEndPosSub.get()
            logger.info("Current robot position: " + str(robotPos))

            angle = robotPos[2]
            original_angle = 180 #??
            
            #converts meter positions to pixel ones
            robotStartPosPixel = convertRobotPoseToPixel((robotPos[0],robotPos[1]), bottom_left_field_position_meters, image_size_meters, field_size_meters,field_size_pixels)
            robotEndPosPixel = convertRobotPoseToPixel((endPos[0], endPos[1]))

            #If the depth camera is activated, we will avoid the field objects as well as robots!
            if isDepthCameraConnected:
                #rotates depth camera to where the robot is looking at (normal bird eye view map has camera in the left middle spot)
                #  /
                #X - - - -  -       X is our camera
                #  \
                bird_eye_view_map = rotate_CV(bird_eye_view_map, angle, cv2.INTER_LINEAR) 
                maxSize = 640
                
                #calculating where the camera position is now in our new rotated map. We will use this to help overlay
                #this map onto the prebaked field map
                camX = np.floor(np.cos((original_angle+angle) * np.pi/180) * maxSize/2) + bird_eye_view_map.shape[1]//2
                camY = -np.floor(np.sin((original_angle+angle) * np.pi/180) * maxSize/2) + bird_eye_view_map.shape[0]//2
                camX = int(camX)
                camY = int(camY)
                logger.info("Got cam position")
            
                #shifts row indicies based off of camera position (i.e. )
                #This makes it such that bird_eye_view_map[rowShifted,colShifted] aligns exactly with the current robot position
                #Essentially we are translating the indicies of the bird eye view map to that on the field map
                rowShifted = row_array - robotStartPosPixel[1] + camY
                colShifted = column_array - robotStartPosPixel[0] + camX #robotStartPosPixel[1] 
                
                rowShifted = rowShifted.astype('int32')
                colShifted = colShifted.astype('int32')
                
                #indicies that lay beyond the map need to be set to 0 to allow the matricies to be combined
                #all invalid indicies will be set to map to the origin of the bird eye view map, which will be guaranteed open/not blocked
                invalid_indices = (rowShifted < 0) | (rowShifted >= bird_eye_view_map.shape[0]) | \
                                (colShifted < 0) | (colShifted >= bird_eye_view_map.shape[1])
                #sets row to origin
                rowShifted[invalid_indices] = 0 
                colShifted[invalid_indices] = 0
                bird_eye_view_map[0,0] = 0 
                
                
                bird_eye_view_map = cv2.dilate(bird_eye_view_map, kernel2, iterations=1).astype(bird_eye_view_map.dtype)
                logger.info("Got final map")
            
                #adds the depth camera output if we have a depth camera
                new_binary_image = binary_image + bird_eye_view_map[rowShifted,colShifted]
            else:
                new_binary_image = binary_image
                
            #final binary image that will be fed into the 
            new_binary_image = np.where(new_binary_image > 0.5, 255, 0).astype(np.uint8) #adds a threshold. Makes sure all values are binary

            if display_debug:
                coloredImg = cv2.cvtColor(new_binary_image.astype('float32'),cv2.COLOR_GRAY2BGR)
            
            logger.info("Starting ASTAR")
            logger.info("Robot start pos: " + str(robotStartPosPixel))
            logger.info("Robot end pos: " + str(robotEndPosPixel))

            depthCameraViewOutput.putFrame(new_binary_image)
            result = AStar.execute(robotStartPosPixel,robotEndPosPixel,new_binary_image, astar_options)
            
            if display_debug:
                for i in range(len(result) -1):
                    # displayImg[convertToImageCordinate(result[i])] = (255,0,255)
                    line = cv2.line(coloredImg, result[i], result[i+1], (255,0,255), 2)
            
            
            logger.info("Sending results")
            logger.debug(result)
            finalResult = []
            if result != "FAILED":
                result = np.array(result).astype(np.float64)
                result = np.flip(result,0) #because AStar reconstruction starts from the end to the start, flip array to be start to end
                logger.info(result)
                result = convertPixelToRobotPose(result)
                finalResult = list(result.ravel())
            
            resultStream.set(finalResult)
            informFinishedCommandSet.set(True)
            logger.info("Finished run")
        
    
#runs at start of raspberry pi
if __name__ == "__main__":
    #gets from pathweaver
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
    run("/mnt/usblog/logs", "/home/hoot/Robotics-AStarPathFinding/logs/", team=1277, image_config=img_config)