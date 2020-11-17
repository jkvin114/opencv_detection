import cv2 as cv
import numpy as np 
img_height=200
def setLabel(image, str, contour):
    (text_width, text_height), baseline = cv.getTextSize(str, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x,y,width,height = cv.boundingRect(contour)
    pt_x = x+int((width-text_width)/2)
    pt_y = y+int((height + text_height)/2)
    cv.rectangle(image, (pt_x, pt_y+baseline), (pt_x+text_width, pt_y-text_height), (200,200,200), cv.FILLED)
    cv.putText(image, str, (pt_x, pt_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, 8)

def checkSize(approx,img_width):
    x_min=img_width+1
    x_max=0
    y_min=img_height+1
    y_max=0

    for i in range(len(approx)):

        if(approx[i][0][0]<x_min): x_min=approx[i][0][0]
        if(approx[i][0][0]>x_max): x_max=approx[i][0][0]

        if(approx[i][0][1]<y_min): y_min=approx[i][0][1]
        if(approx[i][0][1]>y_max): y_max=approx[i][0][1]
    
    return ((x_max-x_min) > img_width*0.8 ) and ((y_max-y_min) > 50 )



img_color = cv.imread('C:\\Users\\win10\\Desktop\\opencv_detection-main\\opencv_detection-main\\crop.jpg', cv.IMREAD_ANYCOLOR)
height,width,channel=img_color.shape
ratio=width/height
img_color = cv.resize(img_color, dsize=( int(img_height*ratio),img_height), interpolation=cv.INTER_AREA)

img_width=img_height*ratio


def nothing(x):
    pass

cv.namedWindow('Binary')
cv.createTrackbar('threshold', 'Binary', 0, 700, nothing)
cv.setTrackbarPos('threshold', 'Binary', 127)

kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])


#blur=cv.medianBlur(img_color,5)

img_gray=cv.filter2D(img_color,-1,kernel_sharpen_1)
cv.imshow('sharpen', img_gray)

img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
#img_gray=cv.equalizeHist(img_gray)


ret,img_binary = cv.threshold(img_gray, 127, 255,cv.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
while(True):
    low = cv.getTrackbarPos('threshold', 'Binary')
   # ret,img_binary = cv.threshold(img_gray, low, 255, cv.THRESH_BINARY_INV) 
    img_binary=cv.Canny(img_gray, low, 70)


    img = cv.dilate(img_binary, kernel, 1)
    img = cv.erode(img, kernel, 1)

    cv.imshow('Binary', img)

    img_result = cv.bitwise_and(img_color, img_color, mask = img_binary)
    cv.imshow('Result', img_result)
    img_result=img
  #esc
    if cv.waitKey(1)&0xFF == 27:
        break

img_binary = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,201,2)

cv.imshow('thr', img_binary)

contours, hierarchy = cv.findContours(img_result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

min_area=img_width*img_height*0.5*0
max_area=img_width*img_height*0.95
for cnt in contours:
    size = len(cnt)

    epsilon = 0.01 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    size = len(approx)

    if checkSize(approx,img_width) and (cv.contourArea(cnt) > min_area) and (cv.contourArea(cnt) < max_area):
        cv.line(img_color, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 3)
        for k in range(size-1):
            cv.line(img_color, tuple(approx[k][0]), tuple(approx[k+1][0]), (0, 255, 0), 3)

    # if cv.isContourConvex(approx):
    #     if size == 4:
    #         setLabel(img_color, "sq", cnt)
       
    #     else:
    #         setLabel(img_color, str(size), cnt)
    # else:
    #     setLabel(img_color, str(size), cnt)

cv.imshow('result', img_color)
cv.waitKey(0)