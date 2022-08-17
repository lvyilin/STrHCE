import numpy as np
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive


class Aircraft(VisionDataset):
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None, download=False):
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))
        self.manufacturer_classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                                      'images_%s_%s.txt' % ('manufacturer', self.split))
        self.family_classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                                'images_%s_%s.txt' % ('family', self.split))
        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes(self.classes_file)
        (_, family_targets, _, _) = self.find_classes(self.family_classes_file)
        (_, manufacturer_targets, _, _) = self.find_classes(self.manufacturer_classes_file)
        samples = self.make_dataset(image_ids, targets, family_targets, manufacturer_targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx):
        path, target, parent_target, pparent_target = self.samples[idx]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.target_transform is not None:
            parent_target = self.target_transform(parent_target)
        if self.target_transform is not None:
            pparent_target = self.target_transform(pparent_target)
        return img, target, parent_target, pparent_target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self, file):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets, family_targets, manufacturer_targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i], family_targets[i], manufacturer_targets[i])
            images.append(item)
        return images
