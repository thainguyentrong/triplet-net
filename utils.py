import os
import tqdm
import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from dataset import EmbeddingDatabase
from config import Hyperparams as hp

def create_embeddings_db(model, use_gpu, save=False):
    eval_images = [hp.image_dir + fname for fname in os.listdir(hp.image_dir)]
    eval_dataset = EmbeddingDatabase(eval_images)
    embeddings = []
    maps = {}
    with torch.no_grad():
        for idx, (img_tensor, fname) in tqdm.tqdm(enumerate(eval_dataset), desc='Creating database'):
            if use_gpu:
                img_tensor = img_tensor.cuda()
            embedding = model(img_tensor)
            embeddings.append(embedding)
            maps[idx] = fname
    if save:
        torch.save({'db_embeddings': torch.stack(embeddings, dim=1).squeeze(dim=0), 'maps': maps}, './database/embedding_vector.pkl')
    return torch.stack(embeddings, dim=1).squeeze(dim=0), maps

def inference(model, img_tensor):
    with torch.no_grad():
        embedding = model(img_tensor)
        embedding = embedding.to(img_tensor.device)
    return embedding

def calculate_AP(prediction, score, groundtruth):
    labels = []
    for fname in prediction:
        if fname in groundtruth:
            labels.append(1)
        else:
            labels.append(0)
    if np.sum(labels) == 0:
        return 0.0
    return average_precision_score(np.array(labels), score)

def sift_detect_and_compute(images, normalize=False, return_keypoints=False, keep_top_k=-1):
    keypoints = np.array([])
    descriptors = np.array([])
    s = cv2.xfeatures2d.SIFT_create()
    for image in images:
        kp = s.detect(image)
        kp, des = s.compute(image, kp)
        if keep_top_k > 0:
            order = np.argsort([-k.response for k in kp])
            kp = [kp[order[j]] for j in range(min(keep_top_k, len(kp)))]
            des = np.array([des[order[j]] for j in range(min(keep_top_k, len(kp)))])
        keypoints = np.concatenate((keypoints, kp), axis=0) if len(keypoints) != 0 else kp
        if des is None or des.shape[0] == 0:
            des = np.zeros((128, 128))
        if normalize:
            des = des / (np.linalg.norm(des, axis=1, keepdims=True) + 1e-15)
        descriptors = np.concatenate((descriptors, des), axis=0) if len(descriptors) != 0 else des

    if return_keypoints:
        return descriptors, keypoints
    else:
        return descriptors.astype(np.float)

def surf_detect_and_compute(images, normalize=False, return_keypoints=False, keep_top_k=-1):
    keypoints = np.array([])
    descriptors = np.array([])
    s = cv2.xfeatures2d.SURF_create(extended=True)
    for image in images:
        kp = s.detect(image)
        kp, des = s.compute(image, kp)
        if keep_top_k > 0:
            order = np.argsort([-k.response for k in kp])
            kp = [kp[order[j]] for j in range(min(keep_top_k, len(kp)))]
            des = np.array([des[order[j]] for j in range(min(keep_top_k, len(kp)))])
        keypoints = np.concatenate((keypoints, kp), axis=0) if len(keypoints) != 0 else kp
        if des is None or des.shape[0] == 0:
            des = np.zeros((128, 128))
        if normalize:
            des = des / (np.linalg.norm(des, axis=1, keepdims=True) + 1e-15)
        descriptors = np.concatenate((descriptors, des), axis=0) if len(descriptors) != 0 else des

    if return_keypoints:
        return descriptors, keypoints
    else:
        return descriptors.astype(np.float)

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram