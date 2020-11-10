import cv2 as cv

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



img_color = cv.imread('C:\\Users\\iu889\\OneDrive\\Desktop\\python\\crop.jpg', cv.IMREAD_ANYCOLOR)
height,width,channel=img_color.shape
ratio=width/height
img_color = cv.resize(img_color, dsize=( int(img_height*ratio),img_height), interpolation=cv.INTER_AREA)

img_width=img_height*ratio


def nothing(x):
    pass

cv.namedWindow('Binary')
cv.createTrackbar('threshold', 'Binary', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'Binary', 127)

img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
cv.imshow('result', img_gray)

ret,img_binary = cv.threshold(img_gray, 127, 255,cv.THRESH_OTSU)

while(True):
    low = cv.getTrackbarPos('threshold', 'Binary')
    ret,img_binary = cv.threshold(img_gray, low, 255, cv.THRESH_BINARY_INV) 

    cv.imshow('Binary', img_binary)

    img_result = cv.bitwise_and(img_color, img_color, mask = img_binary)
    cv.imshow('Result', img_result)
  #esc
    if cv.waitKey(1)&0xFF == 27:
        break


contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    size = len(cnt)

    epsilon = 0.01 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    size = len(approx)

    if(checkSize(approx,img_width)):
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