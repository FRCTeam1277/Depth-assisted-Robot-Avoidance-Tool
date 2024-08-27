import datetime
import logging
from pathlib import Path

import cv2
import depthai as dai
import ntcore
import numpy as np
from cscore import CameraServer

import AStar #should import c library, but will use python as backup
import robotDetection as rd
from AStar import AStarOptions

# Logging Tools
def startLogToUSB(logger, usb_log_file_path, local_log_file_path):
    """Saves logs to a local usb on the raspberry pi to easily debug issues
    If no usb is detected at the location specified, it will save locally on the pi
    Args:
        logger (Logger): Logger that is saving all data
        usbLogFile (str): Location of usb on raspberry pi
        localLogFile (str): Back up location on raspberry pi (local)
    """
    current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    try:
        logging.basicConfig(filename=usb_log_file_path + current_time + ".log",
                            format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.DEBUG)
    except:
        logging.basicConfig(filename=local_log_file_path + current_time + ".log",
                            format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.DEBUG)
        logger.info("USB not detected, saving locally")

# Conversion Tools


def convertRobotPoseToPixel(position_meters, bottom_left_field_meters, image_size_meters, field_size_meters, field_size_pixels):
    """Converts a robot's current position on the field (in meters) into pixel cordiantes on the bird eye view map
    """
    # use top left and right, as well as image size, to determine locations
    position_meters = (position_meters[0] + bottom_left_field_meters[0],  image_size_meters[1] + (
        field_size_meters[1] - (position_meters[1] + bottom_left_field_meters[1])))
    position_pixels = (int(np.round(field_size_pixels[0] * position_meters[0]/field_size_meters[0])), int(
        np.round(field_size_pixels[1] * position_meters[1]/field_size_meters[1])))
    return position_pixels


def convertPixelToRobotPose(position_pixels, bottomLeftFieldMeters, imgSizeMeters, fieldSizeMeters, fieldInPixels):
    pixel_offset = position_pixels * \
        np.array([fieldSizeMeters[0]/(fieldInPixels[0]),
                 fieldSizeMeters[1]/(fieldInPixels[1])])
    pixel_offset = pixel_offset * np.array([1, -1])
    position_meters = pixel_offset + \
        np.array([-bottomLeftFieldMeters[0], imgSizeMeters[1] +
                 fieldSizeMeters[1] - bottomLeftFieldMeters[1]])
    return position_meters


def feetToMeters(cordinates):
    """Converts [x feet,y feet, ...] to meters

    Args:
        cordinates (nd.array): array of floats

    Returns:
        nd.array: [x meters, y meters, ...]
    """
    newTuple = []
    for i in cordinates:
        newTuple.append(i / 3.281)
    return tuple(newTuple)

# https://stackoverflow.com/questions/24318078/is-there-a-faster-way-to-rotate-a-large-rgba-image-than-scipy-interpolate-rotate


def rotateCV(image, angle, interpolation):
    """Rotates an image around its center by an angle (CCW)
    """
    h, w = image.shape[:2]
    cX, cY = (w//2, h//2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h), flags=interpolation)
    return rotated

# Camera functions


def runCamera(camera, depth_camera_config, depth_camera_processor):
    camera.startDepthCamera(depth_camera_config)
    camera.runCamera(depth_camera_processor, depth_camera_config)


# main function
def run(usb_log_path, local_log_path, team, image_field_config, image_scale=0.25, display_debug=False, binary_field_map_local_file_location="2024-field_binary.png", camera_network_table_name="depthcamera"):
    """Runs the main script

    Args:
        usb_log_path (str): path of usb device to save files on
        local_log_path (str): local path if usb log path isn't found
        team (int): FRC team number (used for shuffleboard connection)
        image_field_config (dictionary): image configs from pathweaver
        image_scale (float, optional): How much to scale given field map (lower means more optimized). Defaults to 0.25.
        display_debug (bool, optional): DEPRECATED. Defaults to False.
        binary_field_map_local_file_location (str, optional): Local file path to binary field map. Defaults to "2024-field_binary.png".
        camera_network_table_name (str, optional): Name of the shuffleboard table to send depth camera info to. Defaults to "depthcamera".
    """
    # INTALIZATION STAGE -> SETTING UP ALL VARIABLES TO BE USED IN RUNNING STAGE
    # Setting up logger
    logger = logging.getLogger()
    startLogToUSB(logger, usb_log_path, local_log_path)
    logger.info("DepthAI Script Running (Logging started properly)")

    # getting network tables to communicate with robot
    logger.info("Connecting to network table")
    networktable_instance = ntcore.NetworkTableInstance.getDefault()
    table_depth = networktable_instance.getTable(camera_network_table_name)
    table_robot = networktable_instance.getTable("SmartDashboard")
    networktable_instance.startClient4("depth client")
    networktable_instance.setServerTeam(team)
    logger.info("Connected")

    # tells robot we are connected
    robot_connection_topic = table_depth.getBooleanTopic(
        "Depth Raspberry PI Connected").publish()
    robot_connection_topic.set(True)
    logger.info("Updated connected status")

    # enables camera debugging
    logger.info("Enabling Camera Stream logging")
    CameraServer.enableLogging()
    logger.info("Enabled")

    # provides depth camera output for debugging
    depth_camera_view_output = CameraServer.putVideo("Depth Camera", 640, 480)

    # sets up A* options
    lattice_length = 5
    astar_options = AStarOptions(lattice_length)

    # loads current prebaked binary map of the field
    binary_field_map = Path(__file__).with_name(
        binary_field_map_local_file_location)
    binary_field_map_file_location = str(binary_field_map.absolute())
    logger.info("Opening field map at: " + binary_field_map_file_location)
    binary_image = cv2.imread(
        binary_field_map_file_location, cv2.IMREAD_GRAYSCALE)
    binary_image = cv2.resize(
        binary_image, (0, 0), fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)

    #Todo, deprecated (used when testing on PC and lost when translating code to rpi), either remove or recreate
    if display_debug:
        colored_image_display = cv2.cvtColor(
            binary_image, cv2.COLOR_GRAY2RGB)  # used for debugging

    binary_image_size_pixels = np.asarray(binary_image.shape[::-1])
    field_size_feet = image_field_config["field-size"]
    field_size_meters = feetToMeters(field_size_feet)

    # defining borders (image map is slightly bigger than the actual playing field)
    # Thus we need to add a border where the robot can't be at
    # Also the 0,0 meter point is the bottom left corner of the field, not the 0,0 pixel location, thus we will need to offset this)

    # gets the top left position of the field in pixels (the last place a robot can physically be)
    top_left_field_position_pixels = np.asarray(
        image_field_config["top-left"]) * image_scale  # pixels
    # gets the bottom right position of the field in pixels
    bottom_right_field_position_pixels = np.asarray(
        image_field_config["bottom-right"]) * image_scale  # pixels

    # gets field length in pixels
    field_size_pixels = (bottom_right_field_position_pixels[0] - top_left_field_position_pixels[0],
                         bottom_right_field_position_pixels[1] - top_left_field_position_pixels[1])

    # gets bottom left field position which is (x of top left, y of bottom right) = (left,bottom)
    bottom_left_field_position_meters = ((field_size_meters[0] * top_left_field_position_pixels[0]/field_size_pixels[0]), (
        field_size_meters[1] * bottom_right_field_position_pixels[1]/field_size_pixels[1]))

    # gets scale of how many pixels per meter (we are NOT assuming uniform scaling in both axis, but they should be simialr)
    meters_to_pixel_scale = (
        field_size_pixels[0]/field_size_meters[0], field_size_pixels[1]/field_size_meters[1])

    # gets image size in meters
    image_size_meters = binary_image_size_pixels / meters_to_pixel_scale

    # generates two arrays, going 0,1,2,3,4,5,. See function examples
    # This will be used later on to calculate angles between TODO!!!!
    row_array, column_array = np.indices(binary_image.shape)

    # RUNNING CAMERA
    depth_camera_connected = False
    logger.info("Starting depth camera")
    try:
        depth_camera_config = rd.DepthCameraConfig(
            meters_to_pixel_scale[0], meters_to_pixel_scale[1], 0.5)
        depth_camera = rd.DepthCamera(depth_camera_config)
        depth_camera_processor = rd.DepthCameraProcessing(depth_camera_config)
        depth_camera.startDepthCamera(depth_camera_config)
        depth_camera_connected = True
    except:
        depth_camera_connected = False

    # 30 inches is max robot width, 7 inches is approximate bumper thickness (on both ends), +5 pixels is arbitrary but used for saftey (have some space between the robot and obsticles)
    # Should be around [24,24] -> could prebake this but that saves little time
    robot_half_width_with_bumpers_pixels = np.array(np.floor(meters_to_pixel_scale * np.array(
        # 5 to give more space
        feetToMeters([(30 + 7)/12, (30 + 7)/12])) / 2) + 5, np.uint8)
    robot_padding_kernel = np.ones(
        robot_half_width_with_bumpers_pixels, np.uint8)
    depth_camera_padding_kernel = np.ones((3, 3), np.uint8)

    # since our trajectories are based off of the path of the center of the robot, we need to make sure the robot will be able to
    # drive the path we generate for it
    # Thus we must pad all the borders by half of the robot length to ensure it can get around with the paths we give it
    binary_image = cv2.dilate(binary_image, robot_padding_kernel, iterations=1)

    logger.info("Starting depth camera pipeline")
    if depth_camera_connected:
        try:
            device = dai.Device(depth_camera.pipeline)
        except:
            depth_camera_connected = False

    if depth_camera_connected:
        logger.info("Depth Camera Detected and running smoothly!")
    else:
        logger.info(
            "No depth camera detected, rerun script (limited functionality for AStar without depth input)")

    # Tells network table that we have a feed from the depth camera
    depth_camera_connection_topic = table_depth.getBooleanTopic(
        "depth-camera-connected").publish()
    depth_camera_connection_topic.set(depth_camera_connected)

    # Output queue will be used to get the disparity frames from the outputs defined above
    if depth_camera_connected is True:
        disparityOutputQueue = device.getOutputQueue(
            name="depth", maxSize=1, blocking=False)

    passive_mode_topic = table_depth.getBooleanTopic(
        "passive").subscribe(False)
    passive_mode_on = False
    passive_mode_data_topic = table_depth.getRawTopic(
        "passive-data").publish("byte array")

    run_command_topic_subscriber = table_depth.getBooleanTopic(
        "Trajectory Request").subscribe(False)
    run_command_topic_setter = table_depth.getBooleanTopic(
        "Trajectory Request").publish()
    target_position_subscriber = table_depth.getDoubleArrayTopic(
        "Trajectory End Point").subscribe([0, 0, 0])

    inform_finished_command_setter = table_depth.getBooleanTopic(
        "Trajectory Request Fulfilled").publish()

    result_stream_topic = table_depth.getDoubleArrayTopic(
        "Trajectory Data").publish()
    robot_position_subscriber = table_robot.getDoubleArrayTopic(
        "Position").subscribe([0.0, 0.0, 0.0])

    shuffleboard_update_ticker = 0

    # RUNS A* AND CAMERA LOOP
    logger.info("Starting camera stream")
    while True:
        if depth_camera_connected is True:

            # Start of getting the birds eye view map
            depth_channel = disparityOutputQueue.get()
            # gets numpy array of depths per each pixel
            depth_frame = depth_channel.getFrame()
            logger.info("Processing depth frame")
            raw_bird_eye_view_map = depth_camera_processor.processDepthFrame(
                depth_frame, depth_camera_config)
            logger.info("Got raw bird eye view map, adding to buffer")

            depth_camera_processor.addToBuffer(raw_bird_eye_view_map)
            logger.info("Frame added to buffer, getting product of all frames")

            # Gets bird eye view of what is in front of the robot from the depth camera!
            bird_eye_view_map = depth_camera_processor.getGuaranteedDepth()

            # constantly checks if passive mode is activated by user on the driver station
            shuffleboard_update_ticker += 1
            if shuffleboard_update_ticker > 10:
                shuffleboard_update_ticker = 0
                passive_mode_on = passive_mode_topic.get()

            logger.info("Ran Loop Iteration")

            # Used for debugging, shows binary map to driver station (uses up bandwidth though!)
            if passive_mode_on is True:
                # imgFinalMap = cv2.cvtColor(finalMap.astype('float32'),cv2.COLOR_GRAY2BGR)
                bird_eye_view_map *= 255
                depth_camera_view_output.putFrame(bird_eye_view_map)
                logger.info("sent image to camera server")

        # generates astar path when ordered to
        # to actively avoid obsticles, send run command via network table constantly!
        # format: "RUN ROBOTPOSX ROBOTPOSY HEADING"
        if run_command_topic_subscriber.get() is True:
            # tells drive station that we are initalizing this function
            run_command_topic_setter.set(False)
            inform_finished_command_setter.set(False)
            logger.info("Fetching depth camera data")

            # gets data from network table of where we are and where to go
            # Fromat is: [posx, posy, angle]
            robot_current_position_meters = robot_position_subscriber.get()
            target_position_meters = target_position_subscriber.get()
            logger.info("Current robot position: " +
                        str(robot_current_position_meters))

            angle_degrees = robot_current_position_meters[2]
            original_angle_degrees = 180  # see below. Since our bird eye view has the pinhole on the left looking out to the right, we are at the 180 degree angle by default on the unit circle

            # converts meter positions to pixel ones
            robot_start_position_pixels = convertRobotPoseToPixel(
                (robot_current_position_meters[0], robot_current_position_meters[1]), bottom_left_field_position_meters, image_size_meters, field_size_meters, field_size_pixels)
            robot_end_position_pixels = convertRobotPoseToPixel(
                (target_position_meters[0], target_position_meters[1]))

            # If the depth camera is activated, we will avoid the field objects as well as robots!
            if depth_camera_connected:
                # rotates depth camera to where the robot is looking at (normal bird eye view map has camera in the left middle spot)
                # Or at the 180 degree position on the unit circle looking towards the center of said circle
                #  /
                # X - - - -  -       X is our camera
                #  \
                bird_eye_view_map = rotateCV(
                    bird_eye_view_map, angle_degrees, cv2.INTER_LINEAR)
                maxSize = max(depth_camera_config.camera_width_pixels,
                              depth_camera_config.camera_height_pixels)

                # calculating where the camera position is now in our new rotated map. We will use this to help overlay
                # this map onto the prebaked field map

                # For the math specifically, think of it as selecting the cordinate on the unit circle around the center of the image
                # We are adding two vectors, from the origin to the center of the image, then from the center to the image to a point on the circle
                # The second vector specifically has magnitude being the radius of the circle (maxsize/2)
                cam_x = np.floor(np.cos((original_angle_degrees+angle_degrees)
                                        * np.pi/180) * maxSize/2) + bird_eye_view_map.shape[1]//2
                cam_y = -np.floor(np.sin((original_angle_degrees+angle_degrees)
                                         * np.pi/180) * maxSize/2) + bird_eye_view_map.shape[0]//2
                cam_x = int(cam_x)
                cam_y = int(cam_y)
                logger.info("Got camera position")

                # shifts row indicies based off of camera position
                # This makes it such that bird_eye_view_map[rowShifted,colShifted] aligns exactly with the current robot position
                # Essentially we are translating the indicies of the bird eye view map to that on the field map
                row_shifted = row_array - \
                    robot_start_position_pixels[1] + cam_y
                col_shifted = column_array - \
                    robot_start_position_pixels[0] + \
                    cam_x  # robotStartPosPixel[1]

                row_shifted = row_shifted.astype('int32')
                col_shifted = col_shifted.astype('int32')

                # indicies that lay beyond the map need to be set to 0 to allow the matricies to be combined
                # all invalid indicies will be set to map to the origin of the bird eye view map, which will be guaranteed open/not blocked
                invalid_indices = (row_shifted < 0) | (row_shifted >= bird_eye_view_map.shape[0]) | \
                    (col_shifted < 0) | (
                        col_shifted >= bird_eye_view_map.shape[1])
                # sets row to origin
                row_shifted[invalid_indices] = 0
                col_shifted[invalid_indices] = 0
                bird_eye_view_map[0, 0] = 0

                # Adds padding to objects to account for error
                bird_eye_view_map = cv2.dilate(
                    bird_eye_view_map, depth_camera_padding_kernel, iterations=1).astype(bird_eye_view_map.dtype)
                logger.info("Got final map")

                # adds the depth camera output if we have a depth camera
                new_binary_image = binary_image + \
                    bird_eye_view_map[row_shifted, col_shifted]
            else:
                new_binary_image = binary_image

            # final binary image that will be fed into the
            new_binary_image = np.where(new_binary_image > 0.5, 255, 0).astype(
                np.uint8)  # adds a threshold. Makes sure all values are binary

            if display_debug:
                coloredImg = cv2.cvtColor(
                    new_binary_image.astype('float32'), cv2.COLOR_GRAY2BGR)

            logger.info("Starting ASTAR")
            logger.info(f"Robot start pos: {robot_start_position_pixels}")
            logger.info(f"Robot end pos: {robot_end_position_pixels}")

            # sends the shuffleboard the current binary map
            depth_camera_view_output.putFrame(new_binary_image)
            result = AStar.execute(
                robot_start_position_pixels, robot_end_position_pixels, new_binary_image, astar_options)

            # TODO, deprecated, figure out how to recreate for PC visualization
            if display_debug:
                for i in range(len(result) - 1):
                    # displayImg[convertToImageCordinate(result[i])] = (255,0,255)
                    line = cv2.line(
                        coloredImg, result[i], result[i+1], (255, 0, 255), 2)

            logger.info("Sending results")
            logger.debug(result)
            final_output = []
            if result != "FAILED":
                result = np.array(result).astype(np.float64)
                # because AStar reconstruction starts from the end to the start, flip array to be start to end
                result = np.flip(result, 0)
                logger.info(result)
                result = convertPixelToRobotPose(
                    result, bottom_left_field_position_meters, image_size_meters, field_size_meters, field_size_pixels)
                final_output = list(result.ravel())

            result_stream_topic.set(final_output)
            inform_finished_command_setter.set(True)
            logger.info("Finished run")


# runs during raspberry pi boot sequence (power on)
if __name__ == "__main__":
    # get from pathweaver
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
    ], }
    run("/mnt/usblog/logs", "logs/", team=1277, image_field_config=img_config)
