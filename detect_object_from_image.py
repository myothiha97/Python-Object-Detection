from imageai.Detection import ObjectDetection
import os
import cv2

excution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('yolo.h5')
detector.loadModel()

# input_img = "/home/mthk/Desktop/video-object-detection/images/city-mart.jpg"
# img = cv2.imread(input_img,0)
# # img = cv2.resize(img , (0,0),fx=0.5 , fy=0.5)
# img = cv2.GaussianBlur(img,(11,11),0)
# save_path = "/home/mthk/Desktop/video-object-detection/images/blur_img4.jpg"
save_path = "/home/mthk/Desktop/video-object-detection/out_video/img1.jpg"
# cv2.imwrite(save_path,img)
out_path = "/home/mthk/Desktop/video-object-detection/output_images/out_img9.jpg"

detections = detector.detectObjectsFromImage(
                input_image=save_path,
                output_image_path=out_path,
                minimum_percentage_probability=30
)
for obj in detections:

    print(obj["name"]," : ",obj["percentage_probability"]," : ",obj["box_points"])