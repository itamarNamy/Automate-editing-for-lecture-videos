import cv2
import pafy
import numpy as np
from matplotlib import pyplot as plt
import imutils
from skimage.filters import threshold_local

lena = cv2.imread('lena.png')
def youtube_download(url):
    vPafy = pafy.new(url)
    play = vPafy.getbest(preftype="mp4")
    play.download()
    return play


def face_recogntion(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    recognited_img = img
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = recognited_img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return recognited_img


def put_filter(filt, img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filter = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    fc = img
    for (x, y, w, h) in faces:
        cv2.rectangle(fc, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        resized_filter = cv2.resize(gray_filter, roi_gray.shape)
        print(resized_filter.shape)
    return fc

def video_face_filtering(video_path, img_filter_path):
    cap = cv2.VideoCapture(video_path)
    img_filter = cv2.imread(img_filter_path)
    while True:
        # Take each frame
        _, frame = cap.read()
        if frame is None:
            break
        # face = mt.face_recogntion(frame)
        face = put_filter(img_filter, frame)
        cv2.imshow('frame', face)
        k = cv2.waitKey(27) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


def video_face_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        # Take each frame
        _, frame = cap.read()
        if frame is None:
            break
        face = face_recogntion(frame)
        cv2.imshow('frame', face)
        k = cv2.waitKey(27) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


def video_edit_and_save(video_path, img_filter_path, video_fps):
    #need to do exception handling
    cap = cv2.VideoCapture(video_path)
    img_filter = cv2.imread(img_filter_path)
    fps = video_fps
    _,img = cap.read()
    height, width, layers = img.shape
    size = (width, height)
    #need to check the extension of a video file
    out = cv2.VideoWriter('output3.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    while True:
        # Take each frame
        _, frame = cap.read()
        if frame is None:
            break
        face = put_filter(img_filter, frame)
        #face = face_recogntion(frame)
        out.write(face)
        print('Editing video...')
    cap.release()
    out.release()
    print('Video edited successfully')


def real_time_denoising():
    cap = cv2.VideoCapture(0)

    while True:
        # Take each frame
        _, frame = cap.read()
        if frame is None:
            break
        dst = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        cv2.imshow('original', frame)
        cv2.imshow('denoised', dst)
        k = cv2.waitKey(27) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(1.2*widthA), int(1.2*widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(1.2*heightA), int(1.2*heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (int(round(maxWidth*1.2)), int(round(maxHeight*1.2))))

    # return the warped image
    return warped


def detect_board(img):
    image = img
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 0, 200)

    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        screenCnt = 0
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is approx:
        # show the contour (outline) of the piece of paper
        print("STEP 2: Find contours of paper")
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        # cv2.imshow("Outline", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image, True
    return image, False


def video_face_detection_and_save(video):
    cap = cv2.VideoCapture(video)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('face_detection.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if ret == True:
            frame = face_recogntion(frame)
            # Write the frame into the file 'face_detection.avi'
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

            # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def edge_detector(img):
    img_size = img.shape

    if len(img_size) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 100, 200)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny edge detector'), plt.xticks([]), plt.yticks([])
    plt.show()




def hough_transform_line_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(img, cmap='gray')
    plt.title('line detection'), plt.xticks([]), plt.yticks([])
    plt.show()


def segmentation(img):
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, 450, 290)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    plt.imshow(img), plt.colorbar(), plt.show()


def histogram(img):
    color =('b', 'g', 'r')
    hist = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        print(histr.shape)
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
        hist.append(histr)

    plt.show()
    return hist

def get_frames(video):
    cap = cv2.VideoCapture(video)
    i = 0
    while True:
        ret, frame = cap.read()

        if ret == True:
            # Write the frame into the file 'face_detection.avi'
            cv2.imwrite('pics/frame_' + str(i) + '.png', frame)
            print('saving frame_' + str(i) + '.png')
            i = i+1
            # Display the resulting frame
            # Press Q on keyboard to stop recording
            if i == 1000:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

            # When everything done, release the video capture and video write objects
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] = 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def add_borders(img, topBorderWidth, bottomBorderWidth, leftBorderWidth, rightBorderWidth):
    color_of_border = [185, 125 ,135];
    return cv2.copyMakeBorder(
                 img,
                 topBorderWidth,
                 bottomBorderWidth,
                 leftBorderWidth,
                 rightBorderWidth,
                 cv2.BORDER_CONSTANT,
                 value=color_of_border
              )

def crop_borders(img, border):
    (height, width, channel) = img.shape
    img = img[1:height-border, 1:width-border]
    return img







