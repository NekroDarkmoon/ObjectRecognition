# Imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
from PIL import Image
from utils import transforms as T
import cv2

from train import PennFudanDataset, get_model, get_transform


def get_masks(mask):
    colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask ==1] = colors[np.random.randint(0,10)]
    colored_mask = np.stack([r,g,b], axis=2)
    return colored_mask


def predict(img_path, confidence):
    CLASS_NAMES = ['__background__', 'pedestrian']
    
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image, _ = transform(image, target=None)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.cuda.device('cpu')
    model = get_model(2)
    model.to(device)
    model.load_state_dict(torch.load('maskrcnn_model.h5'))
    print("Model Loaded")

    image = image.to(device)
    model.eval()
    with torch.no_grad():
        pred = model([image])
        print("Prediction complete.")

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()

    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy().astype(int))]
    pred_acc = [i for i in list(pred[0]['scores'].cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_acc = pred_acc[:pred_t+1]

    return masks, pred_boxes, pred_class, pred_acc



def plot_instance(img_path, confidence=0.5, rect_th=2, text_size=0.5, text_th=2 ):
    masks, boxes, pred_cls, pred_acc = predict(img_path, confidence)
    print("Drawing masks")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(masks)):
        rgb_mask = get_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], (0,0,0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        location = list(boxes[i][0])
        location[0] = location[0] + 90

        cv2.putText(img, round(pred_acc[i], 3).astype(str), tuple(location),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    plot_instance('./pedestrian.jpg', 0.7)