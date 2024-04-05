import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CPU to GPU
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

IMAGE_SIZE = 800

img = Image.open('C:/Users/SeoJun/PycharmProjects/capstone/1.jpg')
img = img.resize((IMAGE_SIZE, int(img.height * IMAGE_SIZE / img.width)))

trf = T.Compose([
    T.ToTensor()
])
input_img = trf(img).to(device) # CPU to GPU
out = model([input_img])[0]

codes = [
  Path.MOVETO,
  Path.LINETO,
  Path.LINETO
]
fig, ax = plt.subplots(1)
ax.imshow(img)
THRESHOLD = 0.9  # 해당 정보의 정확도가 90% 이상인 것만 사용
for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
  score = score.detach().cpu().numpy()  # GPU to CPU
  if score < THRESHOLD:
    continue
  box = box.detach().cpu().numpy()  # GPU to CPU
  keypoints = keypoints.detach().cpu().numpy()[:, :2]  # GPU to CPU
  # 사람에 대한 영역을 그리기
  rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='white',
                           facecolor='none')
  ax.add_patch(rect)
  # 왼쪽 팔에 대한 선 그리기
  path = Path(keypoints[5:10:2], codes)
  line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
  ax.add_patch(line)

  # 오른쪽 팔에 대한 선 그리기
  path = Path(keypoints[6:11:2], codes)
  line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
  ax.add_patch(line)

  # 왼쪽 다리에 대한 선 그리기
  path = Path(keypoints[11:16:2], codes)
  line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
  ax.add_patch(line)

  # 오른쪽 다리에 대한 선 그리기
  path = Path(keypoints[12:17:2], codes)
  line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
  ax.add_patch(line)
  # 눈,코,귀는 노란색으로 그리고 팔다리의 시작점과 끝점은 빨간색으로 그리기
  for i, k in enumerate(keypoints):
    if i < 5:
      RADIUS = 5
      FACE_COLOR = 'yellow'
    else:
      RADIUS = 10
      FACE_COLOR = 'red'
    circle = patches.Circle((k[0], k[1]), radius=RADIUS, facecolor=FACE_COLOR)
    ax.add_patch(circle)
plt.show()
