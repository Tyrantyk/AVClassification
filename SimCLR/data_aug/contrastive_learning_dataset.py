from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.DataLoader import AVDRIVEloader


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_transform1(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        #color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        data_transforms = transforms.Compose([#transforms.RandomRotation(degree=180),
                                              transforms.Resize([size, size]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              #transforms.RandomResizedCrop(size=size, ratio=(1,1)),
                                              #transforms.RandomHorizontalFlip(),
                                              #transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              #GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()]),
        return data_transforms
    def get_transform2(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, ratio=(1,1)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor()]),
        return data_transforms

    def get_dataset(self, name, n_views, img):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_transform1(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_transform1(96),
                                                              n_views),
                                                          download=True),

                          'AVDrive': lambda: AVDRIVEloader(img,ContrastiveLearningViewGenerator(
                                                                  self.get_transform1(320),
                                                                  self.get_transform2(320),
                                                                  n_views))}

        try:
            print("get dataset")
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
