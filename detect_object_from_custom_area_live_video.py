from imageai import Detection
import cv2
import matplotlib.pyplot as plt
from defisheye import Defisheye

dtype = 'linear'
format = 'fullframe'
fov = 180
pfov = 120

model = Detection.ObjectDetection()
model.setModelTypeAsYOLOv3()
model.setModelPath('yolo.h5')
# model.setModelPath('resnet50_coco_best_v2.0.1.h5')
model.loadModel()
# custom = model.CustomObjects(person=True)
vid = "/home/mthk/Desktop/video-object-detection/video/NVR_ch17_main_20201212080008_20201212083025_2.mp4"
camera = cv2.VideoCapture(vid)

rect_points = []
drawing = False

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
    # cv2.imshow('img',img)
    # while True:
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #         break
cropped_imgs = []
cropped_pos = []
flag = True
while flag:
    ret, img = camera.read()
    clone_img = img.copy()
    # new_img = img[0:100 , 0:200]
    cv2.namedWindow(winname = "this is for click event")
    cv2.setMouseCallback("this is for click event",draw_rect_with_drag)
    resized = cv2.resize(img,(400,400) , interpolation=cv2.INTER_AREA)
    while True:
        cv2.imshow('this is for click event',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            flag = False
            break

        elif cv2.waitKey(1) & 0xFF == ord('s'):
            if len(rect_points) >= 2:
                roi = clone_img[rect_points[0][1]:rect_points[1][1] , rect_points[0][0]:rect_points[1][0]]
                cropped_pos.append(rect_points)
                cropped_imgs.append(roi)
                print("img saved")
                
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            img = clone_img.copy()

cv2.destroyAllWindows()
    # if cv2.waitKey(3000) or 0xFF == ord('q'):
    #     flag = False

out_img = "./out_video/img1.jpg"
if len(rect_points) >= 2:
# for i in cropped_imgs:
    conn = True
    while conn:
        ret , org_img = camera.read()
        obj = Defisheye(org_img , dtype=dtype , format = format , fov = fov , pfov = pfov)
        obj.convert(out_img)
        org_img = cv2.imread(out_img)
        for index , pos in enumerate(cropped_pos):
            cv2.rectangle(org_img,pos[0],pos[1],(255,0,0),1)
            txt = f'frame {index}'
            cv2.putText(org_img,txt,(pos[0][0],pos[0][1]-10),cv2.FONT_HERSHEY_SIMPLEX ,0.9, (255,255,255),2)
            new_img = org_img[pos[0][1]:pos[1][1] , pos[0][0]:pos[1][0]]
            
        # roi = clone_img[rect_points[0][1]:rect_points[1][1] , rect_points[0][0]:rect_points[1][0]]


            img , preds = model.detectCustomObjectsFromImage(
                                input_image=new_img,
                                input_type= "array",
                                output_type= "array",
                                custom_objects=None,
                                minimum_percentage_probability=70,
                                display_percentage_probability=False,
                                display_object_name=True)
                                
            # cv2.imshow('',img)
            # print(preds)
            if len(preds) >= 1:
                for pred in preds:
                    if pred['name'] == 'person':
                        print(f"Person Detected in frame {index}")
                    else:
                        print(f"{pred['name']} detected in frame {index}")
            else:
                print(f'nothing detected in frame {index}')
                # else:
                #     print('')
            cv2.imshow('',out_img)
            # plt.imshow(org_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                conn = False
                break

camera.release()
cv2.destroyAllWindows()