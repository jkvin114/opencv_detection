import cv2 as cv
import numpy as np 
from scipy.spatial import distance as dist

img_height=300
select_point_size=13  #점 크기
btn_down = False
selected_point=-1   #움직이고 있는 점,아무점도 선택 안했으면 -1
original_countour=[]    #변형하기 전 경계
main_countour=[]        #움직이고 있는 경계
original_img=[]         #자른 원본 이미지


#크기가 일정 이상 되는 경계만 남겨지도록 함
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


#점 움직이는 창 띄움
def get_points(im):

    cv.imshow("Image", im)
    cv.setMouseCallback("Image", mouse_handler)


def mouse_handler(event, x, y, flags, data):
    global btn_down
    global selected_point

    if event == cv.EVENT_LBUTTONUP and btn_down:
        # #if you release the button, finish the line
        btn_down = False

    #마우스 드래그중 
    elif event == cv.EVENT_MOUSEMOVE and btn_down and selected_point!=-1:
        main_countour[selected_point][0][0]=x
        main_countour[selected_point][0][1]=y
        copyimg=original_img.copy()

        #새로운 좌표에 맞춰 다시 경계를 그림
        lin=cv.line(copyimg, tuple(main_countour[0][0]), tuple(main_countour[len(main_countour)-1][0]), (0, 255, 0), 3)  

        for k in range(len(main_countour)-1):
            lin=cv.line(copyimg, tuple(main_countour[k][0]), tuple(main_countour[k+1][0]), (0, 255, 0), 3)

        for k in range(len(main_countour)):
            pt=cv.circle(copyimg,tuple(main_countour[k][0]),select_point_size,(0,0,255),2)

        #thi is just for a ine visualization
        # image = data['im'].copy()
        # cv.line(image, data['lines'][0][0], (x, y), (0,0,0), 1)
        # cv.imshow("Image", image)
        cv.imshow("Image", copyimg)
        

    #클릭시
    elif event == cv.EVENT_LBUTTONDOWN:
        btn_down = True

        #클릭한 좌표가 점이 있는 좌표인지 체크
        selected_point=check_click_coord(x,y)
        print(selected_point)
            

#클릭한 좌표가 점이 있는 좌표인지 체크
def check_click_coord(x,y):

    for i in range(len(main_countour)):
        pointX=main_countour[i][0][0]
        pointY=main_countour[i][0][1]
    
        if(pointX+select_point_size > x and pointX-select_point_size < x) and (pointY+select_point_size > y and pointY-select_point_size < y):
         
            return i
    
    return -1

#getPerspectiveTransform 을 위한 좌표 정렬
def counter_clockwise_sort(points):
    return sorted(points, key=lambda point: point[0] * (-1 if point[1] >= 0 else 1))


img_color = cv.imread('C:\\Users\\win10\\Desktop\\opencv_detection-main\\opencv_detection-main\\s1.jpg', cv.IMREAD_ANYCOLOR)
height,width,channel=img_color.shape
ratio=width/height
#이미지 높이가 항상 일정하게 변환
img_color = cv.resize(img_color, dsize=( int(img_height*ratio),img_height), interpolation=cv.INTER_AREA)

img_width=img_height*ratio


#canny값 조절 창
cv.namedWindow('Binary')
cv.createTrackbar('canny', 'Binary', 0, 700, nothing)
cv.setTrackbarPos('canny', 'Binary', 127)

#shapen 전용 matrix
kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

#관심영역 지정
x,y,w,h = cv.selectROI("Image", img_color, False, False)
img_color=img_color[y:y+h,x:x+w]
original_img=img_color.copy()

#sharpen image
img_gray=cv.filter2D(img_color,-1,kernel_sharpen_1)
# cv.imshow('sharpen', img_gray)

#흑백으로
img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
#img_gray=cv.equalizeHist(img_gray)


kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)

#canny 값 조절
while(True):
    low = cv.getTrackbarPos('canny', 'Binary')
    img_binary=cv.Canny(img_gray, low, 70)


    img = cv.dilate(img_binary, kernel, 1)
    img = cv.erode(img, kernel, 1)

    cv.imshow('Binary', img)

    img_result = cv.bitwise_and(img_color, img_color, mask = img_binary)
    # cv.imshow('Result', img_result)
    img_result=img
  #esc
    if cv.waitKey(1)&0xFF == 27:
        break

# img_binary = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,201,2)

# cv.imshow('thr', img_binary)

#경계선찾기
contours, hierarchy = cv.findContours(img_result, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

min_area=img_width*img_height*0.5*0
max_area=img_width*img_height*0.95

for cnt in contours:
    size = len(cnt)
    
    #경계선을 사각형으로
    epsilon = 0.05 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    size = len(approx)
    #사각형이고 크기가 일정 이상이고 최외곽 경계선 아니면 사진에 경계선 표시
    if size==4 and checkSize(approx,img_width) and (cv.contourArea(cnt) > min_area) and (cv.contourArea(cnt) < max_area) and not(approx[0][0][0]==0 or approx[0][0][1]==0):
        lin=cv.line(img_color, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 3)
        main_countour=approx
        print(approx)

        original_countour=approx
        for k in range(size-1):
            lin=cv.line(img_color, tuple(approx[k][0]), tuple(approx[k+1][0]), (0, 255, 0), 3)
        for k in range(size):
            pt=cv.circle(img_color,tuple(approx[k][0]),select_point_size,(0,0,255),2)


get_points(img_color) #경계 이동 창 띄움
cv.waitKey(0)

# mask=np.zeros((int(img_height),int(img_width)))
# roi=[]
# for i in range(len(main_countour)):
#     roi.append(tuple(main_countour[i][0]))

# cv.fillPoly(mask,[np.array(roi)],1)
# masked=cv.bitwise_and(original_img,original_img,mask=mask)

#3021
arr=[]
for i in range(len(main_countour)):
    arr.append([main_countour[i][0][0],main_countour[i][0][1]])

cnts=counter_clockwise_sort(arr) #경계 좌표 정렬
print(cnts)

#좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
pts1=np.float32([list(cnts[2]),list(cnts[1]),list(cnts[3]),list(cnts[0])])
pts2=np.float32([[0,0],[300,0],[0,100],[300,100]])
m=cv.getPerspectiveTransform(pts1,pts2)

result=cv.warpPerspective(original_img,m,(int(img_width),int(img_height))) #자른 이미지 똑바로보이게 함
cv.imshow('mask', result)
cv.waitKey(0)
