import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
Time_Count = 1
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

        #어깨 기울기222
        inclination = (keypoints[6][1]-keypoints[5][1])/(keypoints[6][0]-keypoints[5][0])

        # 눈에 원 그리기
        cv2.circle(frame, tuple(keypoints[1]), 5, (255, 0, 0), -1)
        cv2.circle(frame, tuple(keypoints[2]), 5, (255, 0, 0), -1)
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
