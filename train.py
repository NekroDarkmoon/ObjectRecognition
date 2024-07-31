# /////////////////////////////////////////////////////////////////////////////
#       Object Identifycation with gluoncv using a pretrained model
# /////////////////////////////////////////////////////////////////////////////

# from matplotlib import pyplot as plt 
# from gluoncv import model_zoo, data, utils

# net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)

# im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
#                           'gluoncv/detection/biking.jpg?raw=true',
#                           path='biking.jpg')

# x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

# ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# width, height = orig_img.shape[1], orig_img.shape[0]
# masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
# orig_img = utils.viz.plot_mask(orig_img, masks)

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1,1,1)
# ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids, class_names=net.classes, ax=ax)
# plt.show()


# /////////////////////////////////////////////////////////////////////////////
#                   Self Training on a pedestrian dataset 
# /////////////////////////////////////////////////////////////////////////////

# Imports
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision
import warnings
import time

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from utils.engine import train_one_epoch, evaluate
from utils import utils
from utils import transforms as T

# Dataset Class
class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        image = Image.open(img_path).convert("RGB")

        mask = np.array(Image.open(mask_path))
        object_ids = np.unique(mask)
        object_ids = object_ids[1:]

        masks = mask == object_ids[:, None, None]

        object_count = len(object_ids)
        bboxes = list()
        for i in range(object_count):
           pos = np.where(masks[i])
           xmin = np.min(pos[1])
           xmax = np.max(pos[1])
           ymin = np.min(pos[0])
           ymax = np.max(pos[0])
           bboxes.append([xmin, ymin, xmax, ymax])

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((object_count, ), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        is_crowd = torch.zeros((object_count, ), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = is_crowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def __len__(self):
        return len(self.imgs)


# Get Tranformations
def get_transform(X):
    transforms = list()
    transforms.append(T.ToTensor())
    if X:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# Setting up the Segemntation Model
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    input_features = model.roi_heads.box_predictor.cls_score.in_features        
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
    
    input_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layers = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(input_features_mask,
                                                       hidden_layers, num_classes)

    return model

def test_accuracy(model, test_loader):
    model.to('cpu')
    model.eval()
    accuracies = []
    count = len(test_loader)

    with torch.no_grad():
        for pos, data in enumerate(test_loader):
            if ((np.random.randint(1,10)) % 2) == 0:
                pass
            else:
                images, _ = data
                prediction = model(images)
                scores = prediction[0]['scores'].cpu().numpy()
                accuracies.extend(scores)
                print(f'Accuracy: {pos}/{count}  - {scores}')

    model.train()
    
    return np.average(accuracies), np.mean(accuracies)


# Main
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    X = PennFudanDataset('PennFudanPed', get_transform(X=True))
    X_test = PennFudanDataset('PennFudanPed', get_transform(X=False))

    indices = torch.randperm(len(X)).tolist()
    X = torch.utils.data.Subset(X, indices[:-50])
    X_test = torch.utils.data.Subset(X_test, indices[-50:])

    X_loader = torch.utils.data.DataLoader(X, batch_size=1, shuffle=True,
                                           num_workers=4, collate_fn=utils.collate_fn)

    X_test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=False,
                                                num_workers=4, collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)

    epoch_count = 10
    for epoch in range(epoch_count):
        train_one_epoch(model, optim, X_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        results = evaluate(model, X_test_loader, device=device)

    print("Saving Model")
    torch.save(model.state_dict(), "maskrcnn_model.h5")
    print("Finished...")


# Hyperparameter Tuning
def finetuning():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    X = PennFudanDataset('PennFudanPed', get_transform(X=True))
    X_test = PennFudanDataset('PennFudanPed', get_transform(X=False))

    indices = torch.randperm(len(X)).tolist()
    X = torch.utils.data.Subset(X, indices[:-50])
    X_test = torch.utils.data.Subset(X_test, indices[-50:])

    X_loader = torch.utils.data.DataLoader(X, batch_size=1, shuffle=True,
                                           num_workers=8, collate_fn=utils.collate_fn)

    X_test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=False,
                                                num_workers=8, collate_fn=utils.collate_fn)

    #  Parameters
    lrs = [0.005, 0.005, 0.05, 0.1]
    momentums = [0.1, 0.3, 0.5, 0.7, 0.9]
    step_sizes = [1,2,3,5,7]
    gammas = [0.1, 0.2, 0.3, 0.5, 0.7] 


    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)

    losses = list()
    val_scores = list()

    start_time = time.time()
    epoch_count = 15
    for epoch in range(epoch_count):
        t_data = train_one_epoch(model, optim, X_loader, device, epoch, print_freq=20)
        losses.append(float(str(t_data.loss)[:7]))
        lr_scheduler.step()
        _, results = evaluate(model, X_test_loader, device=device)
        val_scores.append(results)

    print("--- %s seconds ---" % (float(time.time() - start_time)))

    print("Saving Model")
    torch.save(model.state_dict(), "maskrcnn_model.h5")
    print("Finished...")

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(list(range(epoch_count)), losses, 'r', label="Training Loss")
    plt.plot(list(range(epoch_count)), val_scores, 'b', label="Validation Score")
    plt.title("Training loss and Validation Scores against epoch Count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # print("Saving Model")
    # torch.save(model.state_dict(), "maskrcnn_model.h5")
    print("Finished...")



    
# Init
if __name__ == "__main__":
    # Run this finetuned version
    warnings.filterwarnings("ignore")
    # main()
    # Find Finetuning methods
    finetuning()

