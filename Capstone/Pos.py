import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

cap = cv2.VideoCapture(0)

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
        cv2.polylines(frame, [keypoints[5:10:2]], isClosed=False, color=(0, 0, 255), thickness=2)  # 왼쪽 팔
        cv2.polylines(frame, [keypoints[6:11:2]], isClosed=False, color=(0, 0, 255), thickness=2)  # 오른쪽 팔
        left_shoulder = tuple(keypoints[5])
        right_shoulder = tuple(keypoints[6])
        cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)
        shoulder_length = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        cv2.putText(frame, f'Shoulder Length: {shoulder_length:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # 눈, 코, 귀에 대한 원 그리기
        # for k in keypoints:
        #     cv2.circle(frame, tuple(k), 5, (0, 255, 255), -1)  # 노란색
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
