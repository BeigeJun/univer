import numpy as np
from PIL import Image
import torch
# import torchvision
from torchvision import models
import torchvision.transforms as t
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

IMAGE_SIZE = 800

img_front = Image.open('C:/Users/wns20/Desktop/3.jpg')
img_front = img_front.resize((IMAGE_SIZE, int(img_front.height * IMAGE_SIZE / img_front.width)))
img_side = Image.open('C:/Users/wns20/Desktop/4.jpg')
img_side = img_side.resize((IMAGE_SIZE, int(img_side.height * IMAGE_SIZE / img_side.width)))
image_width, image_height = img_front.size

trf = t.Compose([
    t.ToTensor()
])

codes = [
  Path.MOVETO,
  Path.LINETO,
  Path.LINETO
]


def make_sceleton(img):
    input_img = trf(img).to(device)
    out = model([input_img])[0]
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    key = torch.zeros((17, 3))
    threshold = 0.9
    for box, score, points in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().cpu().numpy()
        if score < threshold:
            continue
        box = box.detach().cpu().numpy()
        points = points.detach().cpu().numpy()[:, :2]

        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='white',
                                 facecolor='none')
        ax.add_patch(rect)
        for i in range(2):
            path = Path(points[5+i:10+i:2], codes)
            line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
            ax.add_patch(line)

        for i in range(2):
            path = Path(points[11+i:16+i:2], codes)
            line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
            ax.add_patch(line)

        for i, k in enumerate(points):
            if i < 5:
                radius = 5
                face_color = 'yellow'
            else:
                radius = 10
                face_color = 'red'
            circle = patches.Circle((k[0], k[1]), radius=radius, facecolor=face_color)
            ax.add_patch(circle)

        key = points
    plt.show()
    return key


def make_3d_sceleton(sceleton_1, sceleton_2):
    make_3d = np.zeros((len(sceleton_1), 3))
    extract_x = sceleton_1[:, 0]
    extract_y = sceleton_1[:, 1]
    extract_z = sceleton_2[:, 0]
    for i in range(len(extract_x)):
        make_3d[i][0] = extract_x[i]
        make_3d[i][1] = extract_z[i]
        make_3d[i][2] = extract_y[i]
    return make_3d


def show_3d_pos(pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    head = (pos[3] + pos[4]) / 2
    ax.scatter(head[0], head[1], head[2], s=250)
    for i in range(5, 7, 1):
        ax.plot(pos[[i, i+2], 0], pos[[i, i+2], 1], pos[[i, i+2], 2], color='blue')
        ax.plot(pos[[i+2, i+4], 0], pos[[i+2, i+4], 1], pos[[i+2, i+4], 2], color='blue')
    for i in range(11, 13, 1):
        ax.plot(pos[[i, i+2], 0], pos[[i, i+2], 1], pos[[i, i+2], 2], color='green')
        ax.plot(pos[[i+2, i+4], 0], pos[[i+2, i+4], 1], pos[[i+2, i+4], 2], color='green')

    ax.plot([pos[5, 0], pos[6, 0]], [pos[5, 1], pos[6, 1]], [pos[5, 2], pos[6, 2]], color='blue')

    shoulder_mid = (pos[5] + pos[6]) / 2
    ax.plot([shoulder_mid[0], head[0]], [shoulder_mid[1], head[1]], [shoulder_mid[2], head[2]], color='blue')

    ax.plot([pos[5, 0], pos[11, 0]], [pos[5, 1], pos[11, 1]], [pos[5, 2], pos[11, 2]], color='green')
    ax.plot([pos[6, 0], pos[12, 0]], [pos[6, 1], pos[12, 1]], [pos[6, 2], pos[12, 2]], color='green')

    ax.plot([pos[11, 0], pos[12, 0]], [pos[11, 1], pos[12, 1]], [pos[11, 2], pos[12, 2]], color='green')

    ax.set_xlabel('Front')
    ax.set_ylabel('Side')
    ax.set_zlabel('Height')
    ax.set_xlim([0, image_width])
    ax.set_ylim([0, image_height])
    ax.set_zlim([image_height, 0])
    plt.show()


Sceleton_1 = make_sceleton(img_front)
Sceleton_2 = make_sceleton(img_side)
make_3d_pos = make_3d_sceleton(Sceleton_1, Sceleton_2)
show_3d_pos(make_3d_pos)


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
