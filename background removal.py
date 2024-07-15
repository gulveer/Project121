# import cv2 to capture videofeed
import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 640
camera.set(3, 640)
camera.set(4, 640)

# loading the mountain image
mountain = cv2.imread('mount everest.jpg')

# resizing the mountain image as 640 X 640
mountain = cv2.resize(mountain, (640, 640))

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # converting the image to HSV for easy processing
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # creating thresholds
        lower_bound = np.array([0, 0, 100])
        upper_bound = np.array([180, 255, 255])

        # thresholding image
        mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

        # inverting the mask
        mask = cv2.bitwise_not(mask)

        # converting mask to 3 channels
        mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # final image
        final_image = np.where(mask_3_channel == 0, frame, mountain)

        # show it
        cv2.imshow('frame', final_image)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()