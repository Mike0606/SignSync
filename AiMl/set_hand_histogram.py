import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 200, 100, 30, 30  # Square size and position
    d = 15  # Distance between squares
    imgCrop = None
    crop = None
    for i in range(8):  # Rows
        for j in range(6):  # Columns
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw squares
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 200  # Reset x after each row
        y += h + d
    return crop

def try_camera_indices():
    for i in range(5):  # Loop through camera indices
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"Camera found at index {i}")
            # Set frame dimensions
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cam
        cam.release()
    print("Error: No available camera found.")
    return None

def get_hand_hist():
    cam = try_camera_indices()
    if cam is None:
        return

    x, y, w, h = 300, 100, 300, 300
    flagPressedS = False
    imgCrop = None
    hist_list = []  # Store multiple histograms

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture image.")
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (800, 600))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if imgCrop is not None:
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                hist_list.append(hist)  # Add to list
        elif keypress == ord('s'):
            flagPressedS = True
            break
        if not flagPressedS:
            imgCrop = build_squares(img)
        if len(hist_list) > 0:
            dst = cv2.calcBackProject([hsv], [0, 1], hist_list[-1], [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)
        cv2.imshow("Set hand histogram", img)

    cam.release()
    cv2.destroyAllWindows()
    
    # Save histograms
    with open("histograms.pkl", "wb") as f:
        pickle.dump(hist_list, f)

get_hand_hist()
