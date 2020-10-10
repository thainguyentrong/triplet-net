import os
import tqdm
import time
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from model import TripletNet
from config import Hyperparams as hp
from utils import create_embeddings_db, inference, calculate_AP
from dataset import image_preprocessing, QueryExtractor

# use_gpu = torch.cuda.is_available()
use_gpu = False

def create_DB(model):
    if os.path.exists('./database/embedding_vector.pkl'):
        if use_gpu:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load('./database/embedding_vector.pkl', map_location=map_location)
        db_embeddings = ckpt['db_embeddings']
        maps = ckpt['maps']
    else:
        db_embeddings, maps = create_embeddings_db(model, use_gpu, save=True)
    return db_embeddings, maps

def visualize(query_name, prediction, score):
    fig = plt.figure()
    ax1_6 = plt.subplot(151)
    ax2 = plt.subplot(252)
    ax3 = plt.subplot(253)
    ax4 = plt.subplot(254)
    ax5 = plt.subplot(255)
    ax7 = plt.subplot(257)
    ax8 = plt.subplot(258)
    ax9 = plt.subplot(259)


    ax1_6.imshow(cv2.cvtColor(cv2.imread(hp.image_dir + query_name), cv2.COLOR_BGR2RGB))
    ax1_6.axis('off')
    ax1_6.set_title('Query Image')
    ax2.imshow(cv2.cvtColor(cv2.imread(hp.image_dir + prediction[0]), cv2.COLOR_BGR2RGB))
    ax2.axis('off')
    ax2.set_title('Similarity: %.4f' % score[0])
    ax3.imshow(cv2.cvtColor(cv2.imread(hp.image_dir + prediction[1]), cv2.COLOR_BGR2RGB))
    ax3.axis('off')
    ax3.set_title('Similarity: %.4f' % score[1])
    ax4.imshow(cv2.cvtColor(cv2.imread(hp.image_dir + prediction[2]), cv2.COLOR_BGR2RGB))
    ax4.axis('off')
    ax4.set_title('Similarity: %.4f' % score[2])
    ax5.imshow(cv2.cvtColor(cv2.imread(hp.image_dir + prediction[3]), cv2.COLOR_BGR2RGB))
    ax5.axis('off')
    ax5.set_title('Similarity: %.4f' % score[3])
    ax7.imshow(cv2.cvtColor(cv2.imread(hp.image_dir + prediction[4]), cv2.COLOR_BGR2RGB))
    ax7.axis('off')
    ax7.set_title('Similarity: %.4f' % score[4])
    ax8.imshow(cv2.cvtColor(cv2.imread(hp.image_dir + prediction[5]), cv2.COLOR_BGR2RGB))
    ax8.axis('off')
    ax8.set_title('Similarity: %.4f' % score[5])
    ax9.imshow(cv2.cvtColor(cv2.imread(hp.image_dir + prediction[6]), cv2.COLOR_BGR2RGB))
    ax9.axis('off')
    ax9.set_title('Similarity: %.4f' % score[6])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print('Use GPU: ', use_gpu)
    model = torch.nn.DataParallel(TripletNet())
    if use_gpu:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
    
    if os.path.exists(hp.logdir + 'model.pkl'):
        if use_gpu:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load(hp.logdir + 'model.pkl', map_location=map_location)
        model.load_state_dict(ckpt['state_dict'])
        print('Restore model')

    model.eval()
    cs_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    db_embeddings, maps = create_DB(model)
    mAP, time_running = [], []

    q_valid = QueryExtractor(dataset=hp.mode, image_dir=hp.image_dir, label_dir=hp.label_dir, subset='valid')

    for q_name, attribute in q_valid.get_queries().items():
        start = time.time()
        bbox, class_idx = attribute[0], attribute[1]
        # create image tensor
        query_img = image_preprocessing(hp.image_dir + q_name)
        query_tensor = torch.FloatTensor(np.transpose(np.expand_dims(query_img, axis=0), axes=[0, 3, 1, 2]))
        # get embedding vector
        if use_gpu:
            query_tensor = query_tensor.cuda()
            cs_func = cs_func.cuda()
        query_embedding = inference(model, query_tensor)
        similarity = cs_func(query_embedding, db_embeddings).topk(len(q_valid.get_groundtruth()[class_idx]))
        prediction = [maps[idx] for idx in similarity[1].cpu().numpy()]
        end = time.time()
        score = similarity[0].cpu().numpy()
        AP = calculate_AP(prediction=prediction, score=score, groundtruth=q_valid.get_groundtruth()[class_idx])
        mAP.append(AP)
        time_running.append(end - start)

        # visualization
        # visualize(q_name, prediction, score)
        # break


    print('mAP: %f' % (np.mean(mAP) * 100))
    print('Time running: %f secs' % np.mean(time_running))