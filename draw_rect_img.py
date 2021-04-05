import cv2
from imageai.Detection import ObjectDetection
import os

excution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('yolo.h5')
detector.loadModel()

img = cv2.imread("./images/city-mart.jpg")


drawing = False
clone_img = img.copy()
rect_points = []

def draw_rect_with_drag(event , x , y , flags,param):
    global drawing , img , rect_points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_points = [(x,y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # cv2.rectangle(img , pt1 = (ix,iy) , pt2 = (x , y), color=(255,0,0),thickness=-1)
        rect_points.append((x,y))
        cv2.rectangle(img , rect_points[0],rect_points[1],(255,0,0),1)
        

cv2.namedWindow(winname = "city_mark")
cv2.setMouseCallback("city_mark",draw_rect_with_drag)

cropped_imgs = []
while True:

    cv2.imshow("city_mark", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif cv2.waitKey(10) & 0xFF == ord('s'):
        if len(rect_points) >= 2:
            roi = clone_img[rect_points[0][1]:rect_points[1][1] , rect_points[0][0]:rect_points[1][0]]
            cropped_imgs.append(roi)
            print("img saved")
            # cv2.imshow('cropped_img' , roi)
            # cv2.waitKey(0)

    elif cv2.waitKey(1) & 0xFF == ord('r'):
        img = clone_img.copy()

print(len(cropped_imgs))

for i in cropped_imgs:
    
    # cv2.imshow('',i)
    img , preds = detector.detectCustomObjectsFromImage(
                        input_image=i,
                        input_type= "array",
                        output_type= "array",
                        custom_objects=None,
                        minimum_percentage_probability=70,
                        display_percentage_probability=False,
                        display_object_name=True)
    # print(i)
    cv2.imshow('',img)
   
    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue


cv2.destroyAllWindows()
# print(chr(27))