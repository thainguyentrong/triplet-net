import os
import sys
import cv2
from functools import partial
import time
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import QueryExtractor, ImageRetrievalDataset, collate_fn, image_preprocessing
from model import TripletNet
from loss import TripletLoss
from config import Hyperparams as hp
from utils import create_embeddings_db, inference, calculate_AP

use_gpu = torch.cuda.is_available()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d or torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

def train():
    q_train = QueryExtractor(dataset=hp.mode, image_dir=hp.image_dir, label_dir=hp.label_dir, subset='train')
    q_valid = QueryExtractor(dataset=hp.mode, image_dir=hp.image_dir, label_dir=hp.label_dir, subset='valid')

    train_dataset = ImageRetrievalDataset(image_dir=hp.image_dir, data_generator=q_train)
    train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, num_workers=hp.num_workers, drop_last=False, shuffle=True, collate_fn=partial(collate_fn, augment=True), pin_memory=use_gpu)

    model = torch.nn.DataParallel(TripletNet())

    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print('Number of parameters: ', count_parameters(model))

    model.apply(weights_init)
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True
    
    curr_epoch = 0
    if os.path.exists(hp.logdir + 'model.pkl'):
        if use_gpu:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load(hp.logdir + 'model.pkl', map_location=map_location)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        curr_epoch = ckpt['global_step']
        print('Restore model')
    
    for epoch in range(curr_epoch+1, hp.epochs):
        model.train()
        losses = []
        optimizer.zero_grad()
        for step, (img_tensor, target_tensor) in enumerate(train_dataloader):
            loss_per_step = []
            start = time.time()
            if use_gpu:
                img_tensor, target_tensor = img_tensor.cuda(), target_tensor.cuda()
            
            for sub_img, sub_target in zip(img_tensor, target_tensor):
                embeddings = model(sub_img).transpose(1, 0) # (D, S)
                loss = criterion(embeddings, sub_target)
                losses.append(loss.item())
                loss_per_step.append(loss.item())
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()

            end = time.time()    
            sys.stdout.write('\rEpoch: %03d, Step: %04d/%d, Loss: %.9f, Time training: %.2f secs' % (epoch, step+1, len(train_dataloader), np.sum(loss_per_step), end-start))

        scheduler.step(np.mean(losses))

        if epoch % hp.display == 0:
            model.eval()
            cs_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            db_embeddings, maps = create_embeddings_db(model, use_gpu)
            mAP = []

            for q_name, attribute in tqdm.tqdm(q_valid.get_queries().items(), desc='Evaluting'):
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
                score = similarity[0].cpu().numpy()
                AP = calculate_AP(prediction=prediction, score=score, groundtruth=q_valid.get_groundtruth()[class_idx])
                mAP.append(AP)

            print('mAP: %f' % (np.mean(mAP) * 100)) 
            ckpt_path = hp.logdir + 'model.pkl'
            torch.save({"state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "global_step": epoch}, ckpt_path)

if __name__ == '__main__':
    np.random.seed(42)
    print('Use GPU: ', use_gpu)
    if use_gpu:
        print('Device name: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    train()
