
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from PIL import Image
import pickle
import torch
import numpy as np

class ObjectDetector():
    def __init__(self):
        self.model = self.get_object_detection_model()
        self.model.eval()
        self.model.to('cuda')
        self.knn = pickle.load(open('data/object_detection_models/knn_colors_aug.pkl', 'rb'))
        self.tranform = torchvision.transforms.ToTensor()

    def get_object_detection_model(self, num_classes=2):
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load('data/object_detection_models/obj_det_cylinder.ckpt'))
        return model

    def apply_nms(self, orig_prediction, iou_thresh=0.3):
        # torchvision returns the indices of the bboxes to keep
        keep = torchvision.ops.nms(orig_prediction['boxes'],
                                   orig_prediction['scores'], iou_thresh)

        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]

        return final_prediction

    def filter_pred(self, pred, img, threshold=0.95, threshold_knn=0.75):
        pred['boxes'] = pred['boxes'].detach().cpu().numpy().tolist()
        pred['scores'] = pred['scores'].detach().cpu().numpy().tolist()
        pred['labels'] = pred['labels'].detach().cpu().numpy().tolist()
        res = {}
        res['boxes'] = []
        res['scores'] = []
        res['labels'] = []
        colors = []
        for idx, score in enumerate(pred['scores']):
            if score > threshold:
                box = pred['boxes'][idx]
                res['boxes'].append(box)
                res['scores'].append(pred['scores'][idx])
                center_x = int((box[2] + box[0]) / 2)
                center_y = int((box[3] + box[1]) / 2)
                colors.append(img[center_y][center_x])
        if len(colors) > 0:
            boxes = []
            scores = []
            labels = []
            labels_tmp = self.knn.predict_proba(colors)
            for idx, lab in enumerate(labels_tmp):
                idx_true = np.argmax(lab)
                if lab[idx_true] >= threshold_knn:
                    boxes.append(res['boxes'][idx])
                    scores.append(res['scores'][idx]*lab[idx_true])
                    labels.append(idx_true)
            res['boxes'] = boxes
            res['scores'] = scores
            res['labels'] = labels

        return res

    def predict(self, images):
        with torch.no_grad():
            # images = imagesT.cpu().numpy()
            # img_trans = [self.tranform(Image.fromarray(img)).to('cuda') for img in images]
            # prediction = self.model(img_trans)
            
            imgs = (images.type(torch.float32) / 255.0).permute(0,3,1,2)
            prediction = self.model(imgs)
            
            nms_prediction = [self.apply_nms(p, iou_thresh=0.2) for p in prediction]
            res = [self.filter_pred(n, images[i].cpu().numpy()) for i, n in enumerate(nms_prediction)]
        return res
