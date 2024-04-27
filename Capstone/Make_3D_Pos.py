import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CPU to GPU
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

IMAGE_SIZE = 800

img1 = Image.open('C:/Users/wns20/Desktop/1.jpg')
img1 = img1.resize((IMAGE_SIZE, int(img1.height * IMAGE_SIZE / img1.width)))
img2 = Image.open('C:/Users/wns20/Desktop/2.jpg')
img2 = img2.resize((IMAGE_SIZE, int(img2.height * IMAGE_SIZE / img2.width)))

trf = T.Compose([
    T.ToTensor()
])

codes = [
  Path.MOVETO,
  Path.LINETO,
  Path.LINETO
]

def Make_sceleton(img):
    input_img = trf(img).to(device)
    out = model([input_img])[0]
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    THRESHOLD = 0.9  # 해당 정보의 정확도가 90% 이상인 것만 사용

    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
      score = score.detach().cpu().numpy()
      if score < THRESHOLD:
        continue
      box = box.detach().cpu().numpy()
      keypoints = keypoints.detach().cpu().numpy()[:, :2]

      rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='white',
                               facecolor='none')
      ax.add_patch(rect)
      for i in range(2):
          path = Path(keypoints[5+i:10+i:2], codes)
          line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
          ax.add_patch(line)

      for i in range(2):
          path = Path(keypoints[11+i:16+i:2], codes)
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

    return keypoints
def Make_3D_sceleton(sceleton_1, sceleton_2):
    Make_3D_pos = torch.zeros((len(sceleton_1),3))
    extract_x = sceleton_1[:, 0].detach().cpu().numpy()
    extract_y = sceleton_1[:, 1].detach().cpu().numpy()
    extract_z = sceleton_2[:, 1].detach().cpu().numpy()
    for i in range(len(extract_x)):
        Make_3D_pos[i][0] = torch.tensor(extract_x[i])
        Make_3D_pos[i][1] = torch.tensor(extract_y[i])
        Make_3D_pos[i][2] = torch.tensor(extract_z[i])
    return Make_3D_pos

def Show_3D_Pos(Pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(Pos[:, 0], Pos[:, 1], Pos[:, 2])

    ax.set_xlabel('Front')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

Sceleton_1 = Make_sceleton(img1)
Sceleton_2 = Make_sceleton(img2)
Maked_3D_Pos = Make_3D_sceleton(Sceleton_1,Sceleton_2)
Show_3D_Pos(Maked_3D_Pos)
# print(Sceleton_2)
# print(Sceleton_2.shape)
# print(Sceleton_1)
# print(Sceleton_1.shape)
