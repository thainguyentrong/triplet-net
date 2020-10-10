import cv2
import os
import tqdm
import time
import numpy as np
import joblib
import torch
from sklearn.cluster import MiniBatchKMeans
from utils import sift_detect_and_compute, build_histogram, calculate_AP, surf_detect_and_compute, surf_detect_and_compute
from dataset import QueryExtractor
from config import Hyperparams as hp

# use_gpu = torch.cuda.is_available()
use_gpu = False

def SIFT_feature_extraction(data_train, n_feature=320):
    descriptors, fnames = [], []
    for fname in tqdm.tqdm(data_train, desc='Feature extracting'):
        img = cv2.imread(hp.image_dir + fname)
        descriptor = sift_detect_and_compute([img], normalize=True, keep_top_k=n_feature)
        descriptors.append(descriptor)
        fnames.append(fname)
    return descriptors, fnames

def SURF_feature_extraction(data_train, n_feature=320):
    descriptors, fnames = [], []
    for fname in tqdm.tqdm(data_train, desc='Feature extracting'):
        img = cv2.imread(hp.image_dir + fname)
        descriptor = surf_detect_and_compute([img], normalize=True, keep_top_k=n_feature)
        descriptors.append(descriptor)
        fnames.append(fname)
    return descriptors, fnames

def clustering(descriptors, n_dim=128):
    kmeans = MiniBatchKMeans(n_clusters=n_dim, batch_size=3*n_dim, verbose=0)
    kmeans.fit(descriptors)
    return kmeans

def create_db_SIFT(data_train):
    descriptors, fnames = SIFT_feature_extraction(data_train)
    kmeans = clustering(np.concatenate(descriptors, axis=0))
    for des, fname in tqdm.tqdm(zip(descriptors, fnames), desc='Creating database'):
        representation = build_histogram(des, kmeans)
        np.save('./database/BoW/SIFT/' + fname[:-4], representation)
    joblib.dump(kmeans, './database/BoW/SIFT/kmeans_trained.pkl')

def create_db_SURF(data_train):
    descriptors, fnames = SURF_feature_extraction(data_train)
    kmeans = clustering(np.concatenate(descriptors, axis=0))
    for des, fname in tqdm.tqdm(zip(descriptors, fnames), desc='Creating database'):
        representation = build_histogram(des, kmeans)
        np.save('./database/BoW/SURF/' + fname[:-4], representation)
    joblib.dump(kmeans, './database/BoW/SURF/kmeans_trained.pkl') 
    
def train_n_eval(feat_type='SIFT'):
    q_valid = QueryExtractor(dataset=hp.mode, image_dir=hp.image_dir, label_dir=hp.label_dir, subset='valid')
    data_valid = [q_name for q_name, _ in q_valid.get_queries().items()]
    data_train = [fname for fname in os.listdir(hp.image_dir) if fname not in data_valid]
    if feat_type == 'SIFT':
        # train and creat database
        if len(os.listdir('./database/BoW/SIFT/')) == 0:
            create_db_SIFT(data_train)
        # evaluate on queries
        eval(feat_type, q_valid)

    elif feat_type == 'SURF':
        # train and creat database
        if len(os.listdir('./database/BoW/SURF/')) == 0:
            create_db_SURF(data_train)
        # evaluate on queries
        eval(feat_type, q_valid)

    else:
        pass

def eval(feat_type, q_valid):
    maps = {}
    db_embeddings, mAP, time_running = [], [], []
    for idx, fname in enumerate(os.listdir('./database/BoW/%s/' % feat_type)):
        if fname.endswith('.npy'):
            embedding = torch.FloatTensor(np.load('./database/BoW/%s/' % (feat_type) + fname))
            db_embeddings.append(embedding)
            maps[idx] = fname[:-4] + '.jpg'
    db_embeddings = torch.stack(db_embeddings, dim=0).squeeze(dim=0)
    cs_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    kmeans = joblib.load('./database/BoW/%s/kmeans_trained.pkl' % feat_type)

    for q_name, attribute in tqdm.tqdm(q_valid.get_queries().items(), desc='Evaluting'):
        bbox, class_idx = attribute[0], attribute[1]
        start = time.time()
        query_img = cv2.imread(hp.image_dir + q_name)
        des = sift_detect_and_compute([query_img], normalize=True, keep_top_k=320)
        representation = build_histogram(des, kmeans)
        query_embedding = torch.FloatTensor(representation).unsqueeze(dim=0)
        if use_gpu:
            query_embedding = query_embedding.cuda()
            db_embeddings = db_embeddings.cuda()
            cs_func = cs_func.cuda()
        similarity = cs_func(query_embedding, db_embeddings).topk(len(q_valid.get_groundtruth()[class_idx]))
        prediction = [maps[idx] for idx in similarity[1].cpu().numpy()]
        end = time.time()
        score = similarity[0].cpu().numpy()
        AP = calculate_AP(prediction=prediction, score=score, groundtruth=q_valid.get_groundtruth()[class_idx])
        mAP.append(AP)
        time_running.append(end-start)

    print('mAP: %f' % (np.mean(mAP) * 100))
    print('Time running: %f secs' % np.mean(time_running))

if __name__ == "__main__":
    # train_n_eval(feat_type='SIFT')
    train_n_eval(feat_type='SURF')
