from defisheye import Defisheye
import cv2

dtype = 'linear'
format = 'fullframe'
fov = 140
pfov = 90


vid = cv2.VideoCapture("/home/mthk/Desktop/video-object-detection/video/NVR_ch17_main_20201212080008_20201212083025_2.mp4")
# out = cv2.VideoWriter('out1.mp4',-1 , 20.0 , (640 , 480))

while True:
    rect , img = vid.read()
    if rect == True:
        obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
        obj.convert('fisheye1.jpg')
        img = cv2.imread('fisheye1.jpg')
        resize = cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
        # out.write(img)
        cv2.imshow('',resize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

vid.release()
cv2.destroyAllWindows()
