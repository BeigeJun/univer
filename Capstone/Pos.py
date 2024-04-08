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
        self.conv4 = nn.Conv2d(in_channels=27, out_channels=36, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(36, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 36)))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

blink_model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
blink_model = torch.load('C:/Users/wns20/PycharmProjects/pythonProject/model.pt')
Time_Count = 1

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    return image

def Eye(image, keypoint, patch_size=28):
    patch_half_size = patch_size // 2
    keypoint_x, keypoint_y = keypoint

    image_np = image.cpu().numpy()  # torch tensor를 numpy 배열로 변환

    if keypoint_x - patch_half_size < 0 or keypoint_x + patch_half_size >= image_np.shape[2] or keypoint_y - patch_half_size < 0 or keypoint_y + patch_half_size >= image_np.shape[1]:
        return None

    patch = image_np[keypoint_y - patch_half_size:keypoint_y + patch_half_size,
            keypoint_x - patch_half_size:keypoint_x + patch_half_size]
    patch = preprocess_image(patch)
    return patch


cap = cv2.VideoCapture(0)
cnt = False
save_data = [0.0, 0.0]
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
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

        # # 사람에 대한 영역을 직사각형으로 그림
        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        # 팔과 다리에 대한 선 그리기
        # cv2.polylines(frame, [keypoints[5:10:2]], isClosed=False, color=(0, 0, 255), thickness=2)  # 왼쪽 팔
        # cv2.polylines(frame, [keypoints[6:11:2]], isClosed=False, color=(0, 0, 255), thickness=2)  # 오른쪽 팔
        left_shoulder = tuple(keypoints[5])
        right_shoulder = tuple(keypoints[6])
        cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)
        shoulder_length = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))

        #어깨 기울기
        inclination = (keypoints[6][1]-keypoints[5][1])/(keypoints[6][0]-keypoints[5][0])

        # 눈에 원 그리기
        cv2.circle(frame, tuple(keypoints[1]), 5, (255, 0, 0), -1)
        cv2.circle(frame, tuple(keypoints[2]), 5, (255, 0, 0), -1)
        Left_Eye = Eye(input_img, keypoints[1], patch_size=28)
        Right_Eye = Eye(input_img, keypoints[2], patch_size=28)
        if Left_Eye == None:
            print("왼쪽 사망")
        if Right_Eye == None:
            print("오른쪽 사망")
        if Left_Eye != None and Right_Eye != None:
            Left_Eye_result = blink_model(Left_Eye)
            Right_Eye_result = blink_model(Right_Eye)
            print("왼쪽 : ",Left_Eye_result,"오른쪽 : ",Right_Eye_result)
        if cnt == False:
            cnt = True
            save_data[0] = shoulder_length
            save_data[1] = inclination
        if save_data[0] - 10 < shoulder_length and save_data[0] + 10 > shoulder_length:
            if save_data[1] - 0.2 < inclination and save_data[1] + 0.2 > inclination:
                Time_Count += 1

        cv2.putText(frame, f'Shoulder Length : {shoulder_length:.2f}, Shoulder inclination: {inclination:.2f} Time : {Time_Count:d}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# keypoint_names = {
#     0: 'nose',
#     1: 'left_eye',
#     2: 'right_eye',
#     3: 'left_ear',
#     4: 'right_ear',
#     5: 'left_shoulder',
#     6: 'right_shoulder',
#     7: 'left_elbow',
#     8: 'right_elbow',
#     9: 'left_wrist',
#     10: 'right_wrist',
#     11: 'left_hip',
#     12: 'right_hip',
#     13: 'left_knee',
#     14: 'right_knee',
#     15: 'left_ankle',
#     16: 'right_ankle',
#     17: 'neck',
#     18: 'left_palm',
#     19: 'right_palm',
#     20: 'spine2(back)',
#     21: 'spine1(waist)',
#     22: 'left_instep',
#     23: 'right_instep'
# }
