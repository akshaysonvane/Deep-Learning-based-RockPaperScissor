import cv2
import numpy as np
import time

import cnn_model as cnn

# Parameters #####################
minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

guess_gesture = False

last_gesture = -1

binary_mode = True
#################################


kernel = np.ones((15, 15), np.uint8)
kernel2 = np.ones((1, 1), np.uint8)
skin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def skin_mask(frame):
    global guess_gesture, last_gesture
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 2)
    roi = frame[y0:y0 + height, x0:x0 + width]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skin_kernel, iterations=1)
    mask = cv2.dilate(mask, skin_kernel, iterations=1)

    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)
    # cv2.imshow("Blur", mask)

    # bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask=mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    retgesture = cnn.guess_gesture(res)
    if last_gesture != retgesture:
        last_gesture = retgesture
        print(cnn.output[last_gesture])
        time.sleep(0.01)

    return res


# %%
def binary_mask(frame):
    global guess_gesture, last_gesture

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 2)
    roi = frame[y0:y0 + height, x0:x0 + width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # blur = cv2.bilateralFilter(roi,9,75,75)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)

    retgesture = cnn.guess_gesture(res)
    if last_gesture != retgesture:
        last_gesture = retgesture
        print(cnn.output[last_gesture])
        time.sleep(0.01)

    return res


def main():
    global guess_gesture, binary_mode, x0, y0, width, height

    # Call CNN model loading callback
    print("Loading default weight file")
    model = cnn.loadCNN(0)

    ## Grab camera input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Feed', cv2.WINDOW_NORMAL)

    # set size as 640x480
    cap.set(3, 640)
    cap.set(4, 480)

    time.sleep(1)

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 3)

        if ret:
            if binary_mode:
                roi = binary_mask(frame)
            else:
                roi = skin_mask(frame)

        cv2.imshow('Feed', frame)
        cv2.imshow('ROI', roi)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        ## Use Esc key to close the program
        if key == 27:
            break

        ## Use b key to toggle between binary threshold or skinmask based filters
        elif key == ord('b'):
            binary_mode = not binary_mode
            if binary_mode:
                print("Binary Threshold filter active")
            else:
                print("SkinMask filter active")

        ## Use g key to start gesture predictions via CNN
        elif key == ord('g'):
            guess_gesture = not guess_gesture
            print("Prediction Mode - {}".format(guess_gesture))


        ## Use i,j,k,l to adjust ROI window
        elif key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5

    # Realse & destroy
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
