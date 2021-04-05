from imageai.Detection import VideoObjectDetection
import os
import cv2


execution_path = os.getcwd()

camera = cv2.VideoCapture(0)
detector = VideoObjectDetection()

# detector.setModelTypeAsRetinaNet()

# detector.setModelPath(os.path.join(execution_path,"resnet50_coco_best_v2.0.1.h5"))
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()

input_video = "/home/mthk/Desktop/video-object-detection/video/Wuhan's wet markets re-open, but customers remain wary.webm"
video_path = detector.detectObjectsFromVideo(
                input_file_path=input_video,
                output_file_path=os.path.join(execution_path , "object_detection_video2"),
                frames_per_second=20,log_progress=True
)

print(video_path)