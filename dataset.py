import os
import cv2
import torch
import random
import numpy as np
from skimage import transform
from skimage.util import random_noise
from torch.utils.data import Dataset
from config import Hyperparams as hp

class QueryExtractor:
    """
        This class extracts all the queries and triplets for dataset
        Eg run:
            > Define directories
                dataset: 'oxford5k' or 'paris'
                image_dir, label_dir: './dataset/oxford5k/image/', './dataset/oxford5k/label/'
                subset: 'train' or 'valid'
            > Create Query extractor object
                q = QueryExtractor(dataset, img_dir, label_dir, subset)
    """
    def __init__(self, dataset, image_dir, label_dir, subset):
        """
            Initialize the QueryExtractor class
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.dataset = dataset

        self.query_paths = sorted([self.label_dir + fname for fname in os.listdir(self.label_dir) if fname.endswith('query.txt')])
        self.query_attr, self.good_names, self.ok_names, self.junk_names = dict(), dict(), dict(), dict()

        positives_same_class = []
        queries_same_class = [[]]

        for q_path in self.query_paths:
            g_path, o_path, j_path = q_path.replace('query', 'good'), q_path.replace('query', 'ok'), q_path.replace('query', 'junk')
            with open(q_path, 'r', encoding='utf-8') as f:
                query_lines = f.readlines()
            for q_line in query_lines:
                # get query name and bounding box
                q_name, bbox = q_line.strip().split()[0], q_line.strip().split()[1:]
                if dataset == 'oxford5k':
                    q_name = q_name.replace('oxc1_', '') + '.jpg'
                else:
                    q_name = q_name + '.jpg'
                self.query_attr[q_name] = [bbox]
                # get list of good files, ok files, junk_files
                good_list, ok_list, junk_list = list(), list(), list()
                # for good files
                with open(g_path, 'r', encoding='utf-8') as f:
                    good_lines = f.readlines()
                for g_line in good_lines:
                    good_list.append(g_line.strip() + '.jpg')
                self.good_names[q_name] = good_list
                # for ok files
                with open(o_path, 'r', encoding='utf-8') as f:
                    ok_lines = f.readlines()
                for o_line in ok_lines:
                    ok_list.append(o_line.strip() + '.jpg')
                self.ok_names[q_name] = ok_list
                # for junk files
                with open(j_path, 'r', encoding='utf-8') as f:
                    junk_lines = f.readlines()
                for j_line in junk_lines:
                    junk_list.append(j_line.strip() + '.jpg')
                self.junk_names[q_name] = junk_list

                # get positives and queries same class
                relevants = self.good_names[q_name] + self.ok_names[q_name]
                if len(positives_same_class) == 0:
                    positives_same_class.append(relevants)
                    queries_same_class[-1] += [q_name]
                else:
                    false_number = 0
                    for positives in positives_same_class:
                        if positives == relevants:
                            queries_same_class[-1] += [q_name]
                        else:
                            false_number += 1
                    if false_number == len(positives_same_class):
                        positives_same_class.append(relevants)
                        queries_same_class.append([q_name])

        # get groundtruth per query
        self.query_groundtruth, self.queries = dict(), dict()
        for class_idx, (queries, positives) in enumerate(zip(queries_same_class, positives_same_class)):
            positives = [fname for fname in positives if fname not in queries]
            # split to training set and validation set
            train_queries, valid_queries = queries[:int(len(queries)*0.6)], queries[int(len(queries)*0.6):]
            if subset == 'train':
                relevant = train_queries + positives
                queries = train_queries
            elif subset == 'valid':
                relevant = queries + positives
                queries = valid_queries
            else:
                relevant = queries + positives
            self.query_groundtruth[class_idx] = relevant
            for q_name in queries:
                self.queries[q_name] = self.query_attr[q_name] + [class_idx]

    def get_groundtruth(self):
        """
            Return dictionary of class index and positives respectively
        """
        return self.query_groundtruth
    
    def get_other_class(self):
        """
            Return list of other images not in any positive class
        """
        all_positives = []
        q = QueryExtractor(dataset=self.dataset, image_dir=self.image_dir, label_dir=self.label_dir, subset=None)
        for class_idx, pos in q.get_groundtruth().items():
            all_positives += pos
        other = [fname for fname in os.listdir(self.image_dir) if fname not in all_positives]
        return other

    def get_negatives(self, positives):
        """
            Return negatives (corresponding to positives)
        """
        images = self.get_other_class()
        for class_idx, pos in self.get_groundtruth().items():
            images += pos
        negatives = [fname for fname in images if fname not in positives]
        return negatives

    def get_queries(self):
        """
            Return dictionary of queries
        """
        return self.queries

    def generate_tuples(self, num_neg=10):
        """
            Generate samples for training stage
            Returns a tuples with each sample format:
                > sample = [query, positive, negative, negative, ...]
                > target = [-1, 1, 0, 0, ...]
        """
        samples, targets = list(), list()
        for q_name, attribute in self.get_queries().items():
            class_idx = attribute[1]
            positives = self.get_groundtruth()[class_idx]
            negatives = self.get_negatives(positives)
            random.shuffle(negatives)

            for k, positive in enumerate(positives):
                samples.append([q_name, positive] + negatives[k*num_neg: k*num_neg + num_neg])
                targets.append([-1, 1] + [0]*num_neg)
        return samples, targets

class ImageRetrievalDataset(Dataset):
    def __init__(self, image_dir, data_generator):
        self.image_dir = image_dir
        self.samples, self.targets = data_generator.generate_tuples()
    
    def __getitem__(self, index):
        samples, targets = self.samples[index], self.targets[index]
        fpaths = [os.path.join(self.image_dir, fname) for fname in samples]
        return fpaths, targets

    def __len__(self):
        return len(self.samples)

class EmbeddingDatabase(Dataset):
    def __init__(self, images):
        self.images = images

    def __getitem__(self, index):
        imgs = []
        img_path = self.images[index]
        img = image_preprocessing(img_path)
        imgs.append(img)
        return torch.FloatTensor(np.transpose(imgs, axes=[0, 3, 1, 2])), img_path.split('/')[-1]

    def __len__(self):
        return len(self.images)

class RandAugmentation:
    def apply(self, img):
        aug_1 = random.choice([self.flip, self.rotate, self.add_noise, self.add_blur])
        aug_2 = random.choice([self.flip, self.rotate, self.add_noise, self.add_blur])
        return aug_2(aug_1(img))

    def flip(self, img):
        return np.fliplr(img)
    
    def rotate(self, img):
        return transform.rotate(img, angle=random.choice(range(-10, 10)))
    
    def add_noise(self, img):
        mode = random.choice(['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle'])
        return random_noise(img, mode=mode)
    
    def add_blur(self, img):
        ksize = random.choice([3, 5, 7, 9])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

def resize(img):
    wTarget, hTarget = hp.size
    h, w, c = img.shape

    factor_x = w / wTarget
    factor_y = h / hTarget
    factor = max(factor_x, factor_y)
    newSize = (min(wTarget, int(w / factor)), min(hTarget, int(h / factor)))
    img = cv2.resize(img, newSize, cv2.INTER_NEAREST)

    target = np.zeros(shape=(hTarget, wTarget, 3), dtype=np.uint8)
    target[int((hTarget-newSize[1])/2): int((hTarget-newSize[1])/2)+newSize[1], int((wTarget-newSize[0])/2): int((wTarget-newSize[0])/2)+newSize[0], :] = img
    return target

def image_preprocessing(fpath):
    img = cv2.imread(fpath)
    img = resize(img)
    return img

def collate_fn(batch, augment=False):
    all_fpaths = [sample[0] for sample in batch]
    all_targets = [sample[1] for sample in batch]
    img_stack, target_stack = [], []
    for sub_fpaths, sub_targets in zip(all_fpaths, all_targets):
        img_tuple = []
        for fpath, _ in zip(sub_fpaths, sub_targets):
            img = image_preprocessing(fpath)
            if augment:
                img = RandAugmentation().apply(img)
            img_tuple.append(img)
        img_stack.append(img_tuple)
        target_stack.append(sub_targets)
    return torch.FloatTensor(np.transpose(img_stack, axes=(0, 1, 4, 2, 3))), torch.LongTensor(target_stack)

# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from functools import partial
#     q_train = QueryExtractor(dataset='oxford5k', image_dir='./dataset/oxford5k/image/', label_dir='./dataset/oxford5k/label/', subset='train')
#     # q_train = QueryExtractor(dataset='paris', image_dir='./dataset/paris/image/', label_dir='./dataset/paris/label/', subset='train')
#     train_dataset = ImageRetrievalDataset(image_dir=hp.image_dir, data_generator=q_train)
#     train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=True, collate_fn=partial(collate_fn, augment=True), pin_memory=False)
#     for step, (img_tensor, target_tensor) in enumerate(train_dataloader):
#         print(img_tensor.size(), target_tensor.size())
