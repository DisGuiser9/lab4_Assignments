import torch
import cv2
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
resnet_model = models.resnet18(pretrained=True)
resnet_model.eval()


def detect_faces(img_path):
    img = cv2.imread(img_path)  # 读取图像
    results = model(img)  # 使用 YOLOv5 进行目标检测
    faces = results.pandas().xywh[0]  # 获取检测结果
    
    # 选择置信度高于 0.5 的人脸
    faces = faces[faces['confidence'] > 0.5]
    
    detected_faces = []
    for _, row in faces.iterrows():
        x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']].values
        face = img[int(y1):int(y2), int(x1):int(x2)]  # 截取人脸区域
        detected_faces.append(face)
    
    return detected_faces

class FaceDataset(Dataset):
    def __init__(self, image_folder, class_labels, transform=None):
        self.image_folder = image_folder
        self.class_labels = class_labels
        self.transform = transform
        self.img_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 获取图片路径
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)  # 读取图片
        
        # 获取标签（假设标签在文件名中，格式为: "image_classname.jpg"）
        label = self.img_paths[idx].split("/")[-1].split("_")[0]
        label = self.class_labels.index(label)  # 将类别名转换为类别编号
        
        # 检测人脸并裁剪
        faces = detect_faces(img_path)
        
        # 随机选择一个人脸进行分类
        if len(faces) > 0:
            face_img = faces[0]  # 选择第一张检测到的人脸
        else:
            face_img = img  # 如果没有检测到人脸，使用原图

        # 转换为 PIL 格式并应用数据增强
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        if self.transform:
            face_pil = self.transform(face_pil)
        
        return face_pil, label

# 数据转换（标准化和调整大小）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 初始化数据集
class_labels = ['CR7', 'Faker', 'KeJie']
train_folder = './train' 
test_folder= './test'

train_dataset = FaceDataset(image_folder=train_folder, class_labels=class_labels, transform=transform)
test_dataset = FaceDataset(image_folder=test_folder, class_labels=class_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_labels))  # 修改最后一层，适应你的分类任务

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%")

# 训练模型
train(model, train_loader, criterion, optimizer, device, epochs=10)

# 推理函数
def test(model, test_loader, device):
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())  # 将预测结果添加到列表中
            all_labels.extend(labels.cpu().numpy())    # 将真实标签添加到列表中

    return all_preds, all_labels


# 测试模型并计算准确率
all_preds, all_labels = test(model, test_loader, device)

# 计算准确率
correct = sum(np.array(all_preds) == np.array(all_labels))
accuracy = correct / len(all_labels) * 100
print(f"Test Accuracy: {accuracy:.2f}%")


def visualize(img_path, predicted_class):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    
    # 在图像中显示预测类别
    plt.imshow(img_rgb)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def visualize_batch(inputs, labels, preds, class_labels):
    # 转换为 numpy 数组
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    
    # 显示图像
    fig, ax = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        img = inputs[i].transpose(1, 2, 0)  # 转换为 HWC 格式
        img = (img * 0.229 + 0.485) * 255  # 反标准化
        img = img.astype(np.uint8)
        
        ax[i].imshow(img)
        ax[i].axis('off')
        ax[i].set_title(f"Pred: {class_labels[preds[i]]}\nTrue: {class_labels[labels[i]]}")
    
    plt.show()

# 示例：在测试时批量显示预测结果
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    visualize_batch(inputs, labels, preds, class_labels)
    break  # 只显示第一批
