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

img_front = Image.open('C:/Users/wns20/Desktop/3.jpg')
img_front = img_front.resize((IMAGE_SIZE, int(img_front.height * IMAGE_SIZE / img_front.width)))
img_side = Image.open('C:/Users/wns20/Desktop/4.jpg')
img_side = img_side.resize((IMAGE_SIZE, int(img_side.height * IMAGE_SIZE / img_side.width)))
image_width, image_height = img_front.size

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
    KEY = torch.zeros((17, 3))
    THRESHOLD = 0.9
    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
        print("!")
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

        for i, k in enumerate(keypoints):
            if i < 5:
                RADIUS = 5
                FACE_COLOR = 'yellow'
            else:
                RADIUS = 10
                FACE_COLOR = 'red'
            circle = patches.Circle((k[0], k[1]), radius=RADIUS, facecolor=FACE_COLOR)
            ax.add_patch(circle)
        print(keypoints.shape)
        KEY = keypoints
    plt.show()
    return KEY

def Make_3D_sceleton(sceleton_1, sceleton_2):
    print(sceleton_1)
    print(sceleton_2)
    Make_3D_pos = np.zeros((len(sceleton_1), 3))
    extract_x = sceleton_1[:, 0]
    extract_y = sceleton_1[:, 1]
    extract_z = sceleton_2[:, 0]
    for i in range(len(extract_x)):
        Make_3D_pos[i][0] = extract_x[i]
        Make_3D_pos[i][1] = extract_z[i]
        Make_3D_pos[i][2] = extract_y[i]
    return Make_3D_pos

def Show_3D_Pos(Pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Pos[:, 0], Pos[:, 1], Pos[:, 2])

    ax.plot(Pos[[5, 7], 0], Pos[[5, 7], 1], Pos[[5, 7], 2], color='blue')
    ax.plot(Pos[[7, 9], 0], Pos[[7, 9], 1], Pos[[7, 9], 2], color='blue')

    ax.plot(Pos[[6, 8], 0], Pos[[6, 8], 1], Pos[[6, 8], 2], color='blue')
    ax.plot(Pos[[8, 10], 0], Pos[[8, 10], 1], Pos[[8, 10], 2], color='blue')

    ax.plot(Pos[[11, 13], 0], Pos[[11, 13], 1], Pos[[11, 13], 2], color='green')
    ax.plot(Pos[[13, 15], 0], Pos[[13, 15], 1], Pos[[13, 15], 2], color='green')

    ax.plot(Pos[[12, 14], 0], Pos[[12, 14], 1], Pos[[12, 14], 2], color='green')
    ax.plot(Pos[[14, 16], 0], Pos[[14, 16], 1], Pos[[14, 16], 2], color='green')

    ax.set_xlabel('Front')
    ax.set_ylabel('Side')
    ax.set_zlabel('Height')
    ax.set_xlim([0, image_width])
    ax.set_ylim([0, image_height])
    ax.set_zlim([image_height, 0])
    plt.show()



Sceleton_1 = Make_sceleton(img_front)
Sceleton_2 = Make_sceleton(img_side)
Maked_3D_Pos = Make_3D_sceleton(Sceleton_1,Sceleton_2)
Show_3D_Pos(Maked_3D_Pos)


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
