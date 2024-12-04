import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path, non_max_suppression_kpt, output_to_keypoint
from utils.plots import plot_one_box, plot_skeleton_kpts
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load

# Função principal
def detect_and_pose(opt):
    # Configurações iniciais
    source, weights, pose_weights, imgsz = opt.source, opt.weights, opt.pose_weights, opt.img_size
    device = select_device(opt.device)
    half = device.type != 'cpu'  # Half precision para GPUs compatíveis

    # Carregar modelos
    detect_model = attempt_load(weights, map_location=device)  # Modelo de detecção
    pose_weights = torch.load(pose_weights, map_location=device)  # Pesos do modelo de pose
    pose_model = pose_weights['model'].float().eval()  # Modelo de pose
    if half:
        detect_model.half()
        pose_model.half().to(device)

    # Configuração de DataLoader
    dataset = LoadImages(source, img_size=imgsz, stride=int(detect_model.stride.max()))

    # Loop sobre imagens
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # Normalização
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inferência do modelo de detecção
        pred = detect_model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0])  # Apenas pessoas (classe 0)

        # Processar detecções
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # Extração da região da pessoa (bounding box)
                    x1, y1, x2, y2 = map(int, xyxy)
                    person_crop = im0s[y1:y2, x1:x2]

                    # Pré-processamento para estimativa de pose
                    pose_img = letterbox(person_crop, 960, stride=64, auto=True)[0]
                    pose_img = transforms.ToTensor()(pose_img)
                    pose_img = torch.tensor(np.array([pose_img.numpy()]))
                    if half:
                        pose_img = pose_img.half().to(device)

                    # Inferência do modelo de pose
                    with torch.no_grad():
                        output, _ = pose_model(pose_img)
                    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=pose_model.yaml['nc'], 
                                                     nkpt=pose_model.yaml['nkpt'], kpt_label=True)
                    keypoints = output_to_keypoint(output)

                    # Desenhar pontos-chave e bounding box na imagem original
                    for idx in range(keypoints.shape[0]):
                        plot_skeleton_kpts(im0s, keypoints[idx, 7:].T, 3)  # Esqueleto
                    plot_one_box(xyxy, im0s, label=f'Person {conf:.2f}', color=(255, 0, 0), line_thickness=2)

        # Salvar ou mostrar imagem final
        cv2.imwrite('output.jpg', im0s)
        cv2.imshow('Result', im0s)
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='Path to detection weights')
    parser.add_argument('--pose-weights', type=str, default='yolov7-w6-pose.pt', help='Path to pose estimation weights')
    parser.add_argument('--source', type=str, default='inference/images', help='Source of input images/videos')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', default='', help='CUDA device or CPU')
    opt = parser.parse_args()
    detect_and_pose(opt)
