from defisheye import Defisheye
import cv2

dtype = 'linear'
format = 'fullframe'
fov = 200
pfov = 100

img = "/home/mthk/Desktop/video-object-detection/images/fisheye-raw.jpg"
img_out = "fisheye-1.jpg"

img1 = cv2.imread(img)
height = img1.shape[0]
width = img1.shape[1]
channel = img1.shape[2]
print(height , width ,channel)
obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
obj.convert(img_out)