from __future__ import absolute_import, division, print_function
import os
import cv2
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as pil

import h5py as h5
import argparse

from models.erfnet import ERFNet
from dataloader.eval.metrics import SegmentationRunningScore
from dataloader.definitions.labels_file import labels_cityscape_seg_train3_eval

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
os.environ['PYTHONHASHSEED'] = '0'
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # Romera
torch.cuda.manual_seed_all(seed)  # Romera
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Demo(object):
    def __init__(self, options):
        self.opt = options
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # Assertions
        assert os.path.isfile(self.opt.image), "Invalid image!"
        self.opt.image.replace('/', os.sep)
        self.opt.image.replace('\\', os.sep)
        self.image_name = self.opt.image.split(os.sep)[-1]

        if self.opt.model_stage == 1:
            assert self.opt.task in {1}, "Invalid task!"
            assert not self.opt.with_weights, "Weights for stage 1 not available"
        elif self.opt.model_stage == 2:
            assert self.opt.task in {1, 2, 12}, "Invalid task!"
        elif self.opt.model_stage == 3:
            assert self.opt.task in {1, 2, 3, 12, 123}, "Invalid task!"

        # Model and task set-up
        self.num_classes_model = {1: 5, 2: 11, 3: 19}[self.opt.model_stage]
        self.task_low, self.task_high = {1: (0, 5), 2: (5, 11), 3: (11, 19), 12: (0, 11), 123: (0, 19)}[self.opt.task]

        # Create a conventional ERFNet
        self.model = ERFNet(self.num_classes_model, self.opt)
        self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Ground truth
        self.metrics = False
        if self.opt.ground_truth:
            assert os.path.isfile(self.opt.ground_truth), "Invalid ground truth!"
            self.metrics = True
            self.num_classes_score = self.task_high - self.task_low
            self.metric_model = SegmentationRunningScore(self.num_classes_score)

        # Output directory
        if self.opt.output_path:
            if not os.path.isdir(self.opt.output_path):
                os.makedirs(self.opt.output_path)
        else:
            self.opt.output_path = os.path.join(self.opt.image.split(os.sep)[:-1])
        image_extension_idx = self.image_name.rfind('.')
        segmentation_name = self.image_name[:image_extension_idx] + \
                            "_seg_stage_{}_task_{}".format(self.opt.model_stage, self.opt.task) + \
                            self.image_name[image_extension_idx:]
        self.output_image = os.path.join(self.opt.output_path, segmentation_name)
        ground_truth_name = self.image_name[:image_extension_idx] + \
                            "_gt_stage_{}_task_{}".format(self.opt.model_stage, self.opt.task) + \
                            self.image_name[image_extension_idx:]
        self.output_gt = os.path.join(self.opt.output_path, ground_truth_name)

        # stdout output
        print("++++++++++++++++++++++ INIT DEMO ++++++++++++++++++++++++")
        print("Image:\t {}".format(self.opt.image))
        print("GT:\t {}".format(self.opt.ground_truth))
        print("Output:\t {}".format(self.opt.output_path))
        print("Stage:\t {}".format(self.opt.model_stage))
        print("Weights: {}".format(self.opt.with_weights))
        print("Task:\t {}".format(self.opt.task))
        print("!!! MIND THAT THE MODELS WERE TRAINED USING AN IMAGE RESOLUTION OF 1024x512px !!!")

        # Class colours
        labels = labels_cityscape_seg_train3_eval.getlabels()
        colors = [(label.trainId - self.task_low, label.color) for label in labels if
                      label.trainId != 255 and label.trainId in range(0, 19)]
        colors.append((255, (0, 0, 0)))  # void class
        self.id_color = dict(colors)
        self.id_color_keys = [key for key in self.id_color.keys()]
        self.id_color_vals = [val for val in self.id_color.values()]


    def _load_model(self):
        """Load model from disk
        """
        path = self.opt.checkpoint_path
        # checkpoint_path = os.path.join("models", "stage_{}".format(self.opt.model_stage))
        #assert os.path.isdir(checkpoint_path), \
        #    "Cannot find folder {}".format(checkpoint_path)

        # path = os.path.join(checkpoint_path, "{}.pth".format("with_weights" if self.opt.with_weights else "wout_weights"))
        model_dict = self.model.state_dict()
        if self.opt.no_cuda:
            pretrained_dict = torch.load(path, map_location='cpu')
        else:
            pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def process_image(self):
        # Required image transformations
        resize_interp = transforms.Resize((512, 1024), interpolation=pil.BILINEAR)
        transformer = transforms.ToTensor()
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # Load Image
        image = cv2.imread(self.opt.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = pil.fromarray(image)
        native_image_size = image.size

        # Transform image
        image = resize_interp(image)
        image = transformer(image)
        image = normalize(image).unsqueeze(0).to(self.device)

        # Process image
        input_rgb = {("color_aug", 0, 0): image}
        output = self.model(input_rgb)

        # Process network output
        pred_seg = output['segmentation_logits'].float()
        pred_seg = pred_seg[:, self.task_low:self.task_high, ...]
        pred_seg = F.interpolate(pred_seg, (native_image_size[1], native_image_size[0]), mode='nearest')
        pred_seg = torch.argmax(pred_seg, dim=1)
        pred_seg = pred_seg.cpu().numpy()

        # Process ground truth
        gt = None
        if self.opt.ground_truth:
            gt = cv2.imread(self.opt.ground_truth, 0)
            gt[gt < self.task_low] = 255
            gt[gt >= self.task_high] = 255
            gt -= self.task_low
            gt[gt == 255 - self.task_low] = 255
            gt = np.expand_dims(gt, 0)

            self.metric_model.update(gt, pred_seg)
            metrics = self.metric_model.get_scores()
            self._save_metrics(metrics)
            print("\n  " + ("{:>8} | " * 2).format("miou", "maccuracy"))
            print(("&{: 8.3f}  " * 2).format(metrics['meaniou'], metrics['meanacc']) + "\\\\")

        # Save prediction to disk
        self._save_pred_to_disk(pred_seg, gt)

        print("\n-> Done!")


    def _save_metrics(self, metrics):
        ''' Save metrics (class-wise) to disk as HDF5 file.
        '''
        save_path = os.path.join(self.opt.output_path, "demo.h5")

        with h5.File(save_path, 'w') as f:
            grp = f

            # Write mean_IoU, mean_acc and mean prec to file / group
            dset = grp.create_dataset('mean_IoU', data=metrics['meaniou'])
            dset.attrs['Description'] = 'See trainIDs for information on the classes'
            dset = grp.create_dataset('mean_recall', data=metrics['meanacc'])
            dset.attrs['Description'] = 'See trainIDs for information on the classes'
            dset.attrs['AKA'] = 'Accuracy -> TP / (TP + FN)'
            dset = grp.create_dataset('mean_precision', data=metrics['meanprec'])
            dset.attrs['Description'] = 'See trainIDs for information on the classes'
            dset.attrs['AKA'] = 'Precision -> TP / (TP + FP)'

            ids = np.zeros(shape=(len(metrics['iou'])), dtype=np.uint32)

            class_iou = np.zeros(shape=(len(metrics['iou'])), dtype=np.float64)
            class_acc = np.zeros(shape=(len(metrics['acc'])), dtype=np.float64)
            class_prec = np.zeros(shape=(len(metrics['prec'])), dtype=np.float64)

            # Disassemble the dictionary
            for key, i in zip(sorted(metrics['iou']), range(len(metrics['iou']))):
                class_iou[i] = metrics['iou'][key]
                class_acc[i] = metrics['acc'][key]
                class_prec[i] = metrics['prec'][key]

            # Create class_id dataset only once in first layer of HDF5 file when in 'w' mode
            dset = f.create_dataset('trainIDs', data=ids)
            dset.attrs['Description'] = 'trainIDs of classes'

            dset = grp.create_dataset('class_IoU', data=class_iou)
            dset.attrs['Description'] = 'See trainIDs for information on the class order'
            dset = grp.create_dataset('class_recall', data=class_acc)
            dset.attrs['Description'] = 'See trainIDs for information on the class order'
            dset.attrs['AKA'] = 'Accuracy -> TP / (TP + FN)'
            dset = grp.create_dataset('class_precision', data=class_prec)
            dset.attrs['Description'] = 'See trainIDs for information on the class order'
            dset.attrs['AKA'] = 'Precision -> TP / (TP + FP)'

    def _save_pred_to_disk(self, pred, gt=None):
        ''' Save a correctly coloured image of the prediction (batch) to disk.
        '''
        pred = pred[0]
        o_size = pred.shape
        single_pred = pred.flatten()

        if gt is not None:
            single_gt = gt[0].flatten()
            single_pred[single_gt == 255] = 255
            single_gt = self._convert_to_colour(single_gt, o_size)
            cv2.imwrite(self.output_gt, single_gt)

        single_pred = self._convert_to_colour(single_pred, o_size)
        cv2.imwrite(self.output_image, single_pred)


    def _convert_to_colour(self, img, o_size):
        ''' Replace trainIDs in prediction with colours from dict, reshape it afterwards to input dimensions and
            convert RGB to BGR to match openCV's colour system.
        '''
        sort_idx = np.argsort(self.id_color_keys)
        idx = np.searchsorted(self.id_color_keys, img, sorter=sort_idx)
        img = np.asarray(self.id_color_vals)[sort_idx][idx]
        img = img.astype(np.uint8)
        img = np.reshape(img, newshape=(o_size[0], o_size[1], 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo options")

    # SYSTEM options
    parser.add_argument("--image",
                        help="path to image that should be passed into the network",
                        type=str)
    parser.add_argument("--task",
                        help="which task to perform (mind the model stage!)",
                        choices=[1, 2, 3, 12, 123],
                        type=int)
    parser.add_argument("--ground_truth",
                        help="path to ground truth of the image (if metrics should be calculated)",
                        type=str)
    parser.add_argument("--model_stage",
                        help="which model to use",
                        type=int,
                        choices=[1, 2, 3])
    parser.add_argument("--with_weights",
                        help="use a model that has been trained using pixel weights",
                        action="store_true")
    parser.add_argument("--output_path",
                        help="path to output directory (optional)",
                        type=str)
    parser.add_argument("--no_cuda",
                        help="if set disables CUDA",
                        action="store_true")
    parser.add_argument("--weights_init",
                        type=str,
                        default="pretrained")
    parser.add_argument("--cluster_mode",
                        type=str,
                        help="name of the cluster",
                        choices=['laptop', 'cluster', 'phoenix'],
                        default=None)
    parser.add_argument("--checkpoint_path",
                        help="path to checkpoint.pth for inference",
                        type=str
    )

    opt = parser.parse_args()
    evaluator = Demo(opt)
    evaluator.process_image()
