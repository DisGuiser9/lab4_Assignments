## Brief Introduction
### Dataset
```
├──train
├──├──1     # CR7 30 imgs
├──├──2     # Faker 20 imgs
├──├──3     # KeJie 10 imgs
├──test
├──├──1     # CR7 20 imgs
├──├──2     # Faker 20 imgs
├──├──3     # KeJie 20 imgs
```

### Code
```python
face_recognition_dl.py    # Algorithm based on Deep Learning(Pytorch)
face_detector = MTCNN()
model = InceptionResnetV1(pretrained='vggface2')
     
face_recognition_cv.py    # Algorithm based on Deep Learning(OpenCV) 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.EigenFaceRecognizer_create()

rename.ipynb              # Rename the images          
```
