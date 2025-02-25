import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

mtcnn_detector = MTCNN()

class FaceDataset(Dataset):
    def __init__(self, data_folder_path, class_labels, transform=None):
        self.data_folder_path = data_folder_path
        self.class_labels = class_labels
        self.transform = transform

        self.img_paths = []
        self.labels = []

        for dir_name in os.listdir(data_folder_path):
            dir_path = os.path.join(data_folder_path, dir_name)
            if os.path.isdir(dir_path):
                if dir_name.isdigit():
                    label = int(dir_name)
                else:
                    label = self.class_labels.index(dir_name)

                for img_name in os.listdir(dir_path):
                    if img_name.endswith(('.jpg', '.png')):
                        img_path = os.path.join(dir_path, img_name)
                        self.img_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)

        faces, _ = mtcnn_detector.detect(img)
        
        if faces is not None and len(faces) > 0:
            x1, y1, x2, y2 = faces[0]
            face = img[int(y1):int(y2), int(x1):int(x2)]
            face_resized = cv2.resize(face, (224, 224))
        else:
            face_resized = cv2.resize(img, (224, 224))

        face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

        if self.transform:
            face_pil = self.transform(face_pil)

        return face_pil, label


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_labels = ['Unknown', 'CR7', 'Faker', 'KeJie']

train_folder = './train'
train_dataset = FaceDataset(data_folder_path=train_folder, class_labels=class_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
model.logits = nn.Linear(model.logits.in_features, len(class_labels))  

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}%")

train(model, train_loader, criterion, optimizer, device, epochs=40)

def test_and_save_images(model, test_loader, device, class_labels, output_dir='./predictions'):
    model.eval()  # Set model to evaluation mode
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    correct = 0  # Initialize counter for correct predictions
    total = 0    # Initialize counter for total predictions
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Count correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Save images based on the original label (not predicted class)
            for i in range(inputs.size(0)):
                # Convert tensor to numpy image
                img = inputs[i].cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                img = (img * 0.229 + 0.485) * 255  # Reverse normalization
                img = img.astype(np.uint8)
                
                original_label = class_labels[labels[i].item()]
                predicted_class_index = preds[i].item()

                # Ensure the predicted class index is within bounds
                if predicted_class_index < len(class_labels):
                    predicted_class = class_labels[predicted_class_index]
                else:
                    predicted_class = "Unknown"  # If out of range, assign to Unknown
                
                label_folder = os.path.join(output_dir, str(original_label))
                if not os.path.exists(label_folder):
                    os.makedirs(label_folder)
                
                # Generate a file name based on both predicted and original labels
                img_filename = os.path.join(label_folder, f"{predicted_class}_{original_label}_{i}.jpg")
                
                # Save the image
                cv2.imwrite(img_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert back to BGR for OpenCV
    
    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save Logs
    with open(f'./{output_dir}/logs.txt', 'a') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")

    return accuracy

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_folder = './test'
test_dataset = FaceDataset(data_folder_path=test_folder, class_labels=class_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

test_accuracy = test_and_save_images(model, test_loader, device, class_labels, output_dir='./deeplearning')
