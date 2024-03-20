from numpy.core.fromnumeric import resize
import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import transforms_ as trans
import torch
from torchvision import transforms
import random
from torchvision.transforms.functional import InterpolationMode


def make_dataset(root1, root2):
    imgs = []
    dirs = os.listdir(root1)
    for file in dirs:
        fname = file.split('.')[0]

        img = os.path.join(root1, "%s.png" % fname)
        mask = os.path.join(root2, "%s-label.png" % fname)
        imgs.append((img, mask))
    return imgs


class LiverDataset(data.Dataset):

    def __init__(self, root1, root2, transform=False, target_transform=None, train_mode=False, smoothing=0,
                 linear_normlization=False):
        imgs = make_dataset(root1, root2)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.train_mode = train_mode
        self.smoothing = smoothing
        self.linear_normlization = linear_normlization

    def __getitem__(self, index):
        x_path, y_path, cls_label = self.imgs[index]
        img_x = Image.open(x_path)
        img_x = img_x.convert('RGB')
        img_y = Image.open(y_path)
        img_y = Image.fromarray(np.uint8(img_y))
        resize_trans = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
        img_x = resize_trans(img_x)
        img_y = resize_trans(img_y)
        if self.train_mode:
            transform_new = transforms.Compose([
                # transforms.CenterCrop(224),
                # transforms.Resize((224,224),interpolation=Image.NEAREST),
                transforms.RandomHorizontalFlip(0.2),
                transforms.RandomVerticalFlip(0.2),
                # transforms.RandomRotation(30), transforms.RandomApply([transforms.Compose([transforms.CenterCrop(
                # 150),transforms.Resize((160,160),interpolation=Image.NEAREST)])], p=0.5)
            ])
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed)  # apply this seed to img
            torch.manual_seed(seed)
            img_x = transform_new(img_x)
            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)
            img_y = transform_new(img_y)
        if self.transform:
            x_transforms = transforms.Compose([
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            ])
            img_x = x_transforms(img_x)
        if self.linear_normlization:
            img_x = np.array(img_x, dtype=float)
            img_x[img_x != -1] = (img_x[img_x != -1] - img_x.min()) / (img_x.max() - img_x.min())

        img_x = self.target_transform(img_x)

        transform_ = transforms.Grayscale(num_output_channels=1)
        img_y = transform_(img_y)

        img_y = np.array(img_y, dtype=float)

        if not self.smoothing:
            img_y[img_y != 0] = 1
        else:
            img_y[img_y != 0] = 1 - self.smoothing
            img_y[img_y == 0] = self.smoothing

        img_name = os.path.basename(x_path).split('.')[0]

        return img_x, img_y, img_name

    def __len__(self):
        return len(self.imgs)
