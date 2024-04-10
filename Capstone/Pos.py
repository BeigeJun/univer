import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(27 * 6 * 3, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 27 * 6 * 3)))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


blink_model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
blink_model = torch.load('C:/Users/wns20/PycharmProjects/pythonProject/model.pt')
Time_Count = 1

def Eye(image, keypoint, w_patch_size=28,h_patch_size=28):
    if image is None:
        print("이미지가 없습니다.")
        return None

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((w_patch_size, h_patch_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    w_patch_half_size = w_patch_size // 2
    h_patch_half_size = h_patch_size // 2
    keypoint_x, keypoint_y = keypoint

    image_width, image_height = image.size

    if keypoint_x - w_patch_half_size < 0 or keypoint_x + w_patch_half_size >= image_width or keypoint_y - h_patch_half_size < 0 or keypoint_y + h_patch_half_size >= image_height:
        return None

    patch = image.crop((keypoint_x - w_patch_half_size, keypoint_y - h_patch_half_size, keypoint_x + w_patch_half_size, keypoint_y + h_patch_half_size))
    patch = transform(patch)

    return patch.to(device)

cap = cv2.VideoCapture(0)
cnt = False
save_data = [0.0, 0.0]
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    image_size = img.size
    # print("Image size:", image_size)
    trf = T.Compose([T.ToTensor()])
    input_img = trf(img).to(device)

    with torch.no_grad():
        out = model([input_img])[0]
    THRESHOLD = 0.9
    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().cpu().numpy()  # GPU to CPU
        if score < THRESHOLD:
            continue
        box = box.detach().cpu().numpy().astype(int)
        keypoints = keypoints.detach().cpu().numpy().astype(int)[:, :2]

        left_shoulder = tuple(keypoints[5])
        right_shoulder = tuple(keypoints[6])
        cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)
        shoulder_length = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))

        inclination = (keypoints[6][1]-keypoints[5][1])/(keypoints[6][0]-keypoints[5][0])

        cv2.circle(frame, tuple(keypoints[1]), 5, (255, 0, 0), -1)
        cv2.circle(frame, tuple(keypoints[2]), 5, (255, 0, 0), -1)
        Left_Eye = Eye(img, keypoints[1], w_patch_size=50, h_patch_size=28)  # 이미지를 PIL 이미지로 전달
        Right_Eye = Eye(img, keypoints[2], w_patch_size=50, h_patch_size=28)

        Left_Eye_Message = ""
        Right_Eye_Message = ""
        if Left_Eye == None:
            Left_Eye_Message = "fail"
        if Right_Eye == None:
            Right_Eye_Message = "fail"
        if Left_Eye != None and Right_Eye != None:
            Left_Eye_result = blink_model(Left_Eye)
            Right_Eye_result = blink_model(Right_Eye)
            if Left_Eye_result[0][0] > Left_Eye_result[0][1]:
                print(Left_Eye_result[0][0], Left_Eye_result[0][1])
                Left_Eye_Message = "Open"
            else:
                print(Left_Eye_result[0][0], Left_Eye_result[0][1])
                Left_Eye_Message = "Close"
            if Right_Eye_result[0][0] > Right_Eye_result[0][1]:
                Right_Eye_Message = "Open"
            else:
                Right_Eye_Message = "Close"
        if cnt == False:
            cnt = True
            save_data[0] = shoulder_length
            save_data[1] = inclination
        if save_data[0] - 10 < shoulder_length and save_data[0] + 10 > shoulder_length:
            if save_data[1] - 0.2 < inclination and save_data[1] + 0.2 > inclination:
                Time_Count += 1

        cv2.putText(frame, f'Shoulder Length : {shoulder_length:.2f}, Shoulder inclination: {inclination:.2f}, Time : {Time_Count:d}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f'Left eye : {Left_Eye_Message}, Right eye : {Right_Eye_Message}',(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        if Left_Eye is not None:
            left_eye_patch = Left_Eye.cpu().numpy()
            left_eye_patch = np.transpose(left_eye_patch, (1, 2, 0))
            left_eye_patch = cv2.cvtColor(left_eye_patch, cv2.COLOR_GRAY2BGR) * 255
            left_eye_patch = cv2.resize(left_eye_patch, (50, 28))
            left_eye_patch = left_eye_patch.astype(np.uint8)

            left_eye_x = keypoints[1][0] - 25
            left_eye_y = keypoints[1][1] - 14

            frame[left_eye_y:left_eye_y + 28, left_eye_x:left_eye_x + 50] = left_eye_patch

            cv2.rectangle(frame, (left_eye_x, left_eye_y), (left_eye_x + 50, left_eye_y + 28), (0, 255, 0), 1)

        if Right_Eye is not None:
            right_eye_patch = Right_Eye.cpu().numpy()
            right_eye_patch = np.transpose(right_eye_patch, (1, 2, 0))
            right_eye_patch = cv2.cvtColor(right_eye_patch, cv2.COLOR_GRAY2BGR) * 255
            right_eye_patch = cv2.resize(right_eye_patch, (50, 28))
            right_eye_patch = right_eye_patch.astype(np.uint8)

            right_eye_x = keypoints[2][0] - 25
            right_eye_y = keypoints[2][1] - 14

            frame[right_eye_y:right_eye_y + 28,
            right_eye_x:right_eye_x + 50] = right_eye_patch

            cv2.rectangle(frame, (right_eye_x, right_eye_y), (right_eye_x + 50, right_eye_y + 28), (0, 255, 0), 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
