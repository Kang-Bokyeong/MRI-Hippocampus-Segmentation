import os
import random

import cv2
import numpy as np
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset


def resize_with_padding(im, desired_size, color=[0, 0, 0]):
    old_size = im.shape[:2]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im, (top, bottom, right, left), (old_size[1], old_size[0])


class ImageDataset(Dataset):
    def __init__(self, image_dir, seg_dir, target_input_size, train=False, scale=0.1, shift=0.1, rotate=20, flip=0.5):
        self.target_input_size = target_input_size
        self.train = train
        self.samples = []

        # 1) read image
        image_dict = {}
        for (path, dir, files) in os.walk(image_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                ext_lower = ext.lower()
                if ext_lower == '.png' or ext_lower == '.jpg' or ext_lower == '.jpeg':
                    image_dict[filename] = os.path.join(path, filename)

        # 2) read segmentation
        seg_dict = {}
        for (path, dir, files) in os.walk(seg_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                ext_lower = ext.lower()
                if ext_lower == '.png' or ext_lower == '.jpg' or ext_lower == '.jpeg':
                    seg_dict[filename] = os.path.join(path, filename)

        # 3) make train data
        for filename, image_path in image_dict.items():
            seg_filename = filename.replace('ACPC', 'L')
            self.samples.append((image_path, seg_dict[seg_filename]))

        # 4) image translation augmentation
        self.aug_translation_seq = iaa.Sequential([
            iaa.Fliplr(flip),
            iaa.Affine(scale={"x": (1.0 - shift, 1.0 + shift), "y": (1.0 - shift, 1.0 + shift)}, translate_percent={"x": (-scale, scale), "y": (-scale, scale)}, rotate=(-rotate, rotate)),
        ])

        # 5) image color augmentation
        self.aug_color_seq = iaa.Sequential([iaa.AddToHueAndSaturation((-50, 50)),
                                             iaa.GaussianBlur(sigma=(0, 2.0))])

        print('Loaded: {}'.format(len(self.samples)))

    def __getitem__(self, index):
        img_path, seg_path = self.samples[index % len(self.samples)]

        # 1) read image & seg file
        image = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if seg_path:
            seg_img = cv2.imread(seg_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.CV_8U)
        else:
            seg_img = np.zeros(image.shape[0:2]).astype('uint8')


        # 2) resize image
        image, (top, bottom, right, left), (ori_w, ori_h) = resize_with_padding(image, self.target_input_size)
        seg_img, _, _ = resize_with_padding(seg_img, int(self.target_input_size))

        if self.train:
            # 3) seg augmentation
            aug_translation_seq = self.aug_translation_seq.to_deterministic()

            # 4) apply image augmentation
            image_aug = self.aug_color_seq.augment_image(image)  # only for image(not seg)
            image_aug = aug_translation_seq.augment_image(image_aug)

            # 5) apply seg augmentation
            seg_img_aug = aug_translation_seq.augment_image(seg_img)
        else:
            image_aug = image
            seg_img_aug = seg_img

        # 6) transform image
        image_aug = (image_aug.astype(np.float32) / 127.5) - 1.0
        image_aug = image_aug.transpose(2, 0, 1)
        image_aug = torch.from_numpy(image_aug)

        # 7) transform seg
        seg_img_aug = (seg_img_aug.astype(np.float32) / 255.)
        seg_img_aug = torch.from_numpy(seg_img_aug)
        seg_img_aug = seg_img_aug.reshape(1, seg_img_aug.shape[0], seg_img_aug.shape[1])  # match for model output

        if self.train:
            return image_aug, seg_img_aug
        else:
            return image_aug, seg_img_aug, img_path, (top, bottom, right, left), (ori_w, ori_h)

    def __len__(self):
        return len(self.samples)
