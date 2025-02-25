import cv2
import os
import numpy as np

def detect_face(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10)

    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
            
        label = int(dir_name)
        subject_dir_path = data_folder_path + "/" + dir_name
        
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue
            
            image_path = subject_dir_path + "/" + image_name
            print("Loaded an image from : ", image_path)
            #read image
            image = cv2.imread(image_path)
            
            #detect face
            face, rect = detect_face(image)
            try:
                face = cv2.resize(face, (200, 300))
            except:
                print("No face found")
                continue
            
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    
    return faces, labels


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(subject_dir_path, face_recognizer, subjects):
    processed_images = []  # 存储所有处理后的图像
    
    # 获取目录下所有文件
    for image_name in os.listdir(subject_dir_path):
        # 跳过隐藏文件和子目录
        if image_name.startswith("."):
            continue
        print("image name:", image_name)
        image_path = os.path.join(subject_dir_path, image_name)
        
        # 读取图像
        test_img = cv2.imread(image_path)
        if test_img is None:
            print(f"无法读取图像：{image_name}")
            continue
            
        img = test_img.copy()
        
        try:
            # 人脸检测和预处理
            face, rect = detect_face(img)
            resized_face = cv2.resize(face, (200, 300))  # 根据模型需求调整尺寸
        except Exception as e:
            print(f"图像 {image_name} 未识别人脸：{str(e)}")
            continue
        
        # 进行预测
        label, confidence = face_recognizer.predict(resized_face)
        label_text = f"{subjects[label]} ({confidence:.2f})"  # 显示置信度
        
        # 绘制标注
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1]-5)
        
        # 将处理后的图像添加到列表
        processed_images.append(img)
    
    return processed_images  # 返回包含所有图像的列表


subjects = ["", "CR7", "Faker", "KeJie"]
print("Preparing data...")
faces, labels = prepare_training_data("./train")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


face_recognizer = cv2.face.EigenFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


print("Predicting images...")

test_imgs_path_1 = "./test/1"
test_imgs_path_2 = "./test/2"
test_imgs_path_3 = "./test/3"

#perform a prediction
predicted_img1 = predict(test_imgs_path_1, face_recognizer, subjects)
predicted_img2 = predict(test_imgs_path_2, face_recognizer, subjects)
predicted_img3 = predict(test_imgs_path_3, face_recognizer, subjects)

print("Prediction complete")

os.makedirs("./opencv/img1", exist_ok=True)
os.makedirs("./opencv/img2", exist_ok=True)
os.makedirs("./opencv/img3", exist_ok=True)

for i in range(len(predicted_img1)):
    cv2.imwrite(f"./opencv/img1/img{i}.jpg", predicted_img1[i])
for i in range(len(predicted_img2)):
    cv2.imwrite(f"./opencv/img2/img{i}.jpg", predicted_img2[i])
for i in range(len(predicted_img3)):
    cv2.imwrite(f"./opencv/img3/img{i}.jpg", predicted_img3[i])

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
