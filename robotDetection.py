#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter


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
        monoLeft.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setCamera("right")

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        depth.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)

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
                depthFrame = inDepth.getFrame()  # gets numpy array of depths per each pixel

                newMap = depthCameraProcessing.processDepthFrame(
                    depthFrame, depthCameraConfig)

                # gets rid of one pixel detections, expands slightly for donut like shapes to be filled in
                # newMap = ndimage.binary_erosion(newMap, iterations=2).astype(newMap.dtype)
                # newMap = ndimage.binary_dilation(newMap, iterations=2).astype(newMap.dtype)

                depthCameraProcessing.addToBuffer(newMap)
                finalMap = depthCameraProcessing.getGuaranteedDepth()
                finalMap = ndimage.binary_dilation(
                    finalMap, iterations=1).astype(finalMap.dtype)

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
    """Settings for the depth camera and for the post processing step
    """

    def __init__(self, meters_to_pixel_scale_x, meters_to_pixel_scale_y, height_from_floor_meters, 
                 camera_width_pixels=640, camera_height_pixels=480, extended_disparity=False, subpixel=True, 
                 lr_check=True, DEBUG=True):
        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        self.extended_disparity = extended_disparity
        # Better accuracy for longer distance, fractional disparity 32-levels:
        self.subpixel = subpixel
        # Better handling for occlusions:
        self.lr_check = lr_check
        # add windows for debugging
        self.DEBUG = DEBUG
        # scale used to convert meters to a pixel array (important to set before as there is rounding error)
        self.meters_to_pixel_scale_x = int(meters_to_pixel_scale_x)
        self.meters_to_pixel_scale_y = int(meters_to_pixel_scale_y)
        # meter cut off, also used for the ceiling
        self.height_from_floor_meters = height_from_floor_meters

        self.camera_width_pixels = camera_width_pixels
        self.camera_height_pixels = camera_height_pixels


class DepthCameraProcessing:
    """Class that will process a raw depth camera frame and return a post processed bird eye view frame
    """

    def __init__(self, depth_camera_config: DepthCameraConfig):
        """Init will prebake a lot of values to save computation time during run time
        """
        # VV all are constants or prebaked constants, so during run time we do minimal computations
        # This saves A LOT of time!! Made the camera go from like 4 fps to 16ish
        self.focal_length = 1.3 * 1e-3  # 1.3 mm
        self.lens_length = (3*1e-9 * depth_camera_config.camera_width_pixels,
                            3*1e-9 * depth_camera_config.camera_height_pixels)
        self.horizontal_pixel_locations = (np.indices((depth_camera_config.camera_height_pixels, depth_camera_config.camera_width_pixels))[1]
                                           - depth_camera_config.camera_width_pixels/2 * np.ones((depth_camera_config.camera_height_pixels, depth_camera_config.camera_width_pixels))) * 3e-6
        self.horizontal_thetas = np.arctan(
            self.horizontal_pixel_locations / self.focal_length)
        self.tan_horizontal_thetas = np.tan(
            self.horizontal_thetas)  # could be optimized
        self.tan_horizontal_thetas_squared = self.tan_horizontal_thetas * \
            self.tan_horizontal_thetas
        self.normal_distance = np.sqrt(1 + self.tan_horizontal_thetas_squared)
        self.vertical_pixel_locations = -1 * (np.indices((depth_camera_config.camera_height_pixels, depth_camera_config.camera_width_pixels))[
                                              0] - depth_camera_config.camera_height_pixels/2 * np.ones((depth_camera_config.camera_height_pixels, depth_camera_config.camera_width_pixels))) * 3e-6
        self.vertical_thetas = np.arctan(self.vertical_pixel_locations/np.sqrt(
            self.horizontal_pixel_locations * self.horizontal_pixel_locations + self.focal_length * self.focal_length))
        self.tan_vertical_thetas = np.tan(self.vertical_thetas)

        self.buffer_size = 5
        self._buffer_index = 0
        self.image_buffer_array = [None] * (self.buffer_size)

    def processDepthFrame(self, depth_frame_pixels, depth_camera_config: DepthCameraConfig):
        """Creates an initial birds eye view map based off of a depth frame

        Args:
            depth_frame_pixels ([int,int, float]): Depth camera frame (POV is the camera)
            depth_camera_config (DepthCameraConfig): config of our camera

        Returns:
            [int,int,bool]: raw depth image (better to use getGuaranteedDepth though!)
        """
        depth_frame_meters = depth_frame_pixels / \
            1000  # converts depth from mm to meters
        height_from_floor_meters = depth_camera_config.height_from_floor_meters
        # how much to scale on birds eye view image
        scale_x = depth_camera_config.meters_to_pixel_scale_x
        # how much to scale on birds eye view image
        scale_y = depth_camera_config.meters_to_pixel_scale_y

        relative_distance_from_camera = depth_frame_meters  # how far away from camera
        relative_perpendicular_distance_from_camera = depth_frame_meters * \
            self.tan_horizontal_thetas  # how much to the side
        # np.sqrt((relativeDistFromCamera * relativeDistFromCamera) + (relativePerpendicularFromCamera * relativePerpendicularFromCamera)) #distance from camera focal point
        distance = depth_frame_meters * self.normal_distance

        # gaussian_filter(depthCalc[depthCalc > 4], sigma=3)
        depth_frame_meters[depth_frame_meters > 4] = 0

        # creating new image but bird eye view
        new_map = np.empty((depth_camera_config.camera_height_pixels,
                            depth_camera_config.camera_width_pixels))

        # transformations between old and new x and ys
        new_x = relative_distance_from_camera
        # +480/scale puts on the left side center
        new_y = (relative_perpendicular_distance_from_camera) + \
            depth_camera_config.camera_height_pixels/scale_y

        # height relative to camera (offset to the y axis)
        height = distance * self.tan_vertical_thetas

        # removes floor and ceiling from birdseye view
        new_x[height > 1 - height_from_floor_meters] = 0
        new_y[height > 1 - height_from_floor_meters] = 0
        new_x[height < -height_from_floor_meters] = 0
        new_y[height < -height_from_floor_meters] = 0

        new_x = (new_x.ravel() * scale_x).astype(np.int32)
        new_y = (new_y.ravel() * scale_y).astype(np.int32)

        # newX = np.round(newX * scaleX).ravel().astype(np.int32) #converts 2d array of newX cordinates to a 1D array, used to tranverse the newMap array one
        # newY = np.round(newY * scaleY).ravel().astype(np.int32) #^^^

        new_x[new_x < 0] = 0
        new_x[new_x >= depth_camera_config.camera_width_pixels] = 0
        new_y[new_y >= depth_camera_config.camera_height_pixels] = 0
        new_y[new_y < 0] = 0

        # if something was detected, set this pixel to white
        new_map[new_y, new_x] = 1

        return new_map

    def addToBuffer(self, new_frame):
        """Add a raw depth image to our buffer
        """
        buffer_index = self._buffer_index
        self.image_buffer_array[buffer_index] = new_frame.copy()
        self._buffer_index = buffer_index + 1
        if buffer_index >= self.buffer_size - 1:
            self._buffer_index = 0

    def getGuaranteedDepth(self):
        """Gets a final map that is a product of all 5 images in the buffer
        This ensures that if an object is detected, it must have been detected in all 5 images before
        Other wise it is not reported.
        Makes our results more guaranteed
        """
        final_image = self.image_buffer_array[0].copy()
        # return imageBufferArray[bufferIndex]
        for i in range(self.buffer_size - 1):
            if isinstance(self.image_buffer_array[1+i], np.ndarray):
                # if detected, we multiply by 1, if not, by 0
                final_image = final_image * self.image_buffer_array[1+i]
        return final_image


if __name__ == "__main__":

    depthCameraConfig = DepthCameraConfig(170//4, 170//4, 0.5)

    depthCamera = DepthCamera(depthCameraConfig)
    depthCameraProcessing = DepthCameraProcessing(depthCameraConfig)

    depthCamera.startDepthCamera(depthCameraConfig)
    depthCamera.runCamera(depthCameraProcessing, depthCameraConfig)
