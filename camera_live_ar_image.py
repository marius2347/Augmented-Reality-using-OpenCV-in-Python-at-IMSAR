# OpenCV Augmented Reality (AR) with live camera feed

# python packages
import numpy as np # numerical processing
import cv2 # opencv
import sys # exit the script
import argparse # parsing command line arguments

# construct the argument parser and parse the args
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
    help="path to input source image that will be put on input")
args = vars(ap.parse_args())

# load the source image
source = cv2.imread(args["source"])

# open the camera
cap = cv2.VideoCapture(4)

# check if camera opened successfully
if not cap.isOpened():
    print("[INFO] Could not open camera. Exiting!")
    sys.exit(0)

# load the ArUCo dictionary and parameters
print("[INFO] Detecting markers...")
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters()

while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Failed to grab frame. Exiting!")
        break

    # resize the frame
    frame = cv2.resize(frame, (600, 600))
    (image_height, image_width) = frame.shape[:2]

    # detect markers in the frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    # print detected marker IDs
    if ids is not None:
        print(f"[INFO] Detected marker IDs: {ids}")

    # need to apply augmented reality if it finds 4 corner markers
    if ids is None or len(corners) != 4:
        print("[INFO] Could not find 4 corners. Showing original frame.")
        cv2.imshow("AR Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # found the 4 ArUco markers, flatten the IDs list & initialize reference points
    ids = ids.flatten()
    refPts = []

    # loop over the IDs of the ArUco markers in top-left, top-right, bottom-right, bottom-left
    for i in (923, 1001, 241, 1007):
        j = np.squeeze(np.where(ids == i))
        if j.size == 0:
            print(f"[INFO] Could not find marker ID {i}. Exiting!")
            sys.exit(0)
        corner = np.squeeze(corners[j])
        refPts.append(corner)

    # reference points: top-left, top-right, bottom-right, bottom-left
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = np.array([refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]])

    # grab the spatial dimensions of the source and the transform matrix for the source image
    (srcH, srcW) = source.shape[:2]
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    # compute the homography matrix and then warp the source image to the destination based on the homography
    (H, _) = cv2.findHomography(srcMat, dstMat)

    imgW = image_width
    imgH = image_height
    warped = cv2.warpPerspective(source, H, (imgW, imgH))

    # construct a mask for the source image now that the perspective warp has taken place 
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)

    # this step is optional, but to give the source image a black border surrounding it when applied to the source image, you can apply a dilation operation
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)

    # create a three-channel version of the mask by stacking it depth-wise, such that we can copy the warped source image into the input image
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)

    # copy the warped source image into the input image by (1) multiplying
    # the warped image and masked together, (2) multiplying the original
    # input image with the mask (giving more weight to the input where
    # there *ARE NOT* masked pixels), and (3) adding the resulting
    # multiplications together
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    frameMultiplied = cv2.multiply(frame.astype(float), 1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, frameMultiplied)
    output = output.astype("uint8")

    # show the input frame, source image, output of our augmented reality
    cv2.imshow("AR Output", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
