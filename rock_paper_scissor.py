import cv2
import numpy as np
import tensorflow as tf

model=tf.keras.models.load_model("keras_model.h5")

cam=cv2.VideoCapture(0)
while True:
    ret,frame=cam.read()
    img=cv2.resize(frame,(224,224))
    image=np.array(img,dtype=np.float32)
    image=np.expand_dims(image,axis=0)
    image=image/255.0
    result=model.predict(image)
    print(result)

    cv2.imshow("result",frame)
    if cv2.waitKey(12)==32:
        break
cam.release()
