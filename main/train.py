
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from vqa_bias.cfvqa import CFVQA
from vqa_bias.cfvqa_criterion import CFVQACriterion
from causal_effect_intervention.counterfactual_sample import counterfactual_sample
import _pickle as cPickle
from bleu import bleu


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp
# Train phase
def train(args, cfg, model, train_loader, eval_loader, device, dataset, s_epoch=0):
    #生成日志
    tblog_dir = os.path.join(cfg.OUTPUT_DIR, args.fusion_mode, "tensorboardlogs")
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    writer = SummaryWriter(log_dir=tblog_dir)

    #学习率设置
    if args.fusion_mode == 'normal':
        if 'CLEF2019' in cfg.DATASET.DATASET:
            lr_default = cfg.TRAIN.OPTIMIZER.BASE_LR
            lr_decay_step = 2
            lr_decay_rate = .75
            lr_decay_epochs = range(15, 200, lr_decay_step)
            gradual_warmup_steps = []
        else:
            lr_default = cfg.TRAIN.OPTIMIZER.BASE_LR
            lr_decay_rate = 1
            lr_decay_epochs = []
            gradual_warmup_steps = []
    elif args.fusion_mode == 'cf_hm' or args.fusion_mode == 'cf_sum' or args.fusion_mode == 'cf_mask':

        lr_default = 5e-4
        lr_decay_step = 2
        lr_decay_rate = .50
        lr_decay_epochs = range(35, 200, lr_decay_step)
        gradual_warmup_steps = [5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,
                                3.75e-4,3.75e-4,3.75e-4,3.75e-4,3.75e-4,3.75e-4,3.75e-4,3.75e-4,3.75e-4,3.75e-4,
                                2.81e-4,2.81e-4,2.81e-4,2.81e-4,2.81e-4,2.81e-4,2.81e-4,2.81e-4,2.81e-4,2.81e-4,
                                2.11e-4,2.11e-4,2.11e-4,2.11e-4,2.11e-4,]
    elif args.fusion_mode == 'rubi':
        lr_default = cfg.TRAIN.OPTIMIZER.BASE_LR
        lr_decay_step = 20
        lr_decay_rate = .70
        lr_decay_epochs = range(10, 200, lr_decay_step)
        gradual_warmup_steps = []
    else:
        raise ValueError(f"fusion_mode is not supported!")


    # 模型送到GPU
    model = model.to(device)

    # create packet for output
    utils.create_dir(cfg.OUTPUT_DIR)
    # for every train, create a packet for saving .pth and .log

    ckpt_path = os.path.join(cfg.OUTPUT_DIR, args.fusion_mode)
    utils.create_dir(ckpt_path)
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)



    best_train_score = 0
    epoch_lr = lr_default
    # Epoch passing in training phase
    for epoch in range(s_epoch, cfg.TRAIN.N_EPOCH):
        total_loss = 0
        train_score = 0
        if epoch < len(gradual_warmup_steps):
            epoch_lr = gradual_warmup_steps[epoch]
            logger.info('-------[Epoch]:{}-------'.format(epoch))
            logger.info('gradual warm up lr: {:.8f}'.format(epoch_lr))
        elif epoch in lr_decay_epochs:
            epoch_lr = epoch_lr * lr_decay_rate
            logger.info('-------[Epoch]:{}-------'.format(epoch))
            logger.info('decreased lr: {:.8f}'.format(epoch_lr))
        else:
            logger.info('-------[Epoch]:{}-------'.format(epoch))
            logger.info('lr: {:.8f}'.format(epoch_lr))


        # Adam
        optim = torch.optim.Adamax(params=model.parameters(), lr = epoch_lr)

        number = 0
        model.train()
        
        # Predicting and computing score
        for i, (v, q, a, _, _, _) in enumerate(train_loader):
            optim.zero_grad()
            v_causal_non = v
            v_causal = v # v_causal具有因果关系的反事实样例 v_causal_non不具有因果关系的反事实样例
            if cfg.TRAIN.VISION.MAML:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if cfg.TRAIN.VISION.CLIP:
                if cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                    v[2] = v[2].reshape(v[2].shape[0], 3, 288, 288)
                    v_causal[2] = v_causal[2].reshape(v_causal[2].shape[0], 3, 288, 288)
                    v_causal[1] = counterfactual_sample(v_causal[1], args.k, "causal")
                    v_causal[2] = counterfactual_sample(v_causal[2], args.k, "causal")
                    v_causal_non[2] = v_causal_non[2].reshape(v_causal_non[2].shape[0], 3, 288, 288)
                    v_causal_non[1] = counterfactual_sample(v_causal_non[1], args.k, "non_causal")
                    v_causal_non[2] = counterfactual_sample(v_causal_non[2], args.k, "non_causal")
                else:
                    v[2] = v[2].reshape(v[2].shape[0], 3, 250, 250)
                    v_causal[2] = v_causal[2].reshape(v_causal[2].shape[0], 3, 250, 250)
                    v_causal[1] = counterfactual_sample(v_causal[1], args.k, "causal")
                    v_causal[2] = counterfactual_sample(v_causal[2], args.k, "causal")
                    v_causal_non[2] = v_causal_non[2].reshape(v_causal_non[2].shape[0], 3, 250, 250)
                    v_causal_non[1] = counterfactual_sample(v_causal_non[1], args.k, "non_causal")
                    v_causal_non[2] = counterfactual_sample(v_causal_non[2], args.k, "non_causal")
                v[2] = v[2].to(device)
            if cfg.TRAIN.VISION.OTHER_MODEL:
                v = v.to(device)

            q[0] = q[0].to(device)
            if cfg.LOSS.LOSS_TYPE == "BCELogits":
                label = a.to(device)
            elif cfg.LOSS.LOSS_TYPE == "CrossEntropy":
                label = a.to(device, dtype=torch.int64) #交叉熵要求preds类型为float, label类型为long
            else:
                raise ValueError(f"{cfg.LOSS.LOSS_TYPE} loss not supported!")


            # cf_vqa and criterion
            cf_vqa = CFVQA(args, cfg, model, args.fusion_mode,dataset)
            cf_vqa_out = cf_vqa(v, q)
            cf_vqa_out_causal = cf_vqa(v_causal, q)
            f_vqa_out_causal_non = cf_vqa(v_causal_non, q)
            cf_vqa_criterion = CFVQACriterion(args, cfg, model, label, args.question_loss_weight, args.vision_loss_weight)
            loss = cf_vqa_criterion(cfg, cf_vqa_out, v) - cf_vqa_criterion(cfg, cf_vqa_out_causal, v_causal) + cf_vqa_criterion(cfg, f_vqa_out_causal_non, v_causal_non)
            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            # compute the acc for open and close
            total_loss += loss.item()

            if args.fusion_mode == 'normal' or args.fusion_mode == 'rubi':
                batch_score= compute_score_with_logits(cf_vqa_out['logits_vq'], label.data)
            elif args.fusion_mode == 'cf_hm' or args.fusion_mode == 'cf_sum' or args.fusion_mode == 'cf_mask':
                batch_score= compute_score_with_logits(cf_vqa_out['logits_cfvqa'], label.data)
            else:
                raise ValueError(f"fusion_mode is not supported!")



            train_score += batch_score.sum()
            number += q[0].shape[0]


       
        total_loss /= len(train_loader)
        train_score = 100 * train_score / number
        logger.info('[Train] Loss:{:.6f}, Train_Acc:{:.6f}%'.format(total_loss, train_score))
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Accuracy/train", train_score, epoch)

        if epoch % 5 == 0 and eval_loader is not None:
            eval_score = evaluate_classifier(args, dataset, model, eval_loader, cfg, device)

            logger.info('[validation] Val_Acc:{:.6f}%'.format(eval_score))


        if epoch == cfg.TRAIN.N_EPOCH - 1:
            model_path = os.path.join(ckpt_path, 'best_model.pth')
            utils.save_model(model_path, model, epoch, optim)

        # if train_score > best_train_score:
        #     best_train_score = train_score
        #     best_epoch = epoch
        #     model_path = os.path.join(ckpt_path, 'best_model.pth')
        #     utils.save_model(model_path, model, best_epoch, optim)



        
# Evaluation
def evaluate_classifier(args, dataset, model, dataloader, cfg, device):
    score = 0
    model.eval()
    with torch.no_grad():
        for i,(v, q, a,_, _, _, _, _, _) in enumerate(dataloader):
            if cfg.TRAIN.VISION.MAML:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if cfg.TRAIN.VISION.CLIP:
                if cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                    v[2] = v[2].reshape(v[2].shape[0], 3, 288, 288)
                else:
                    v[2] = v[2].reshape(v[2].shape[0], 3, 250, 250)
                v[2] = v[2].to(device)
            if cfg.TRAIN.VISION.OTHER_MODEL:
                v = v.to(device)
            
            q[0] = q[0].to(device)
            if cfg.TRAIN.QUESTION.CLIP:
                q[1] = q[1].to(device)
            label = a.to(device) #(test_batchSize, 1024)

            if args.fusion_mode == 'rubi' or args.fusion_mode == 'normal':
                q_embedding, _ = model(args, v, q)
                preds = model.classifier(q_embedding)  #(test_batchSize, 1024)
            elif args.fusion_mode == 'cf_hm' or args.fusion_mode == 'cf_sum' or args.fusion_mode == 'cf_mask':
                cf_vqa = CFVQA(args, cfg, model, args.fusion_mode, dataset)
                cf_vqa_out = cf_vqa(v, q)
                preds = cf_vqa_out['logits_cfvqa']
            else:
                raise ValueError(f"fusion model is not supported!")

            batch_score = compute_score_with_logits(preds, label.data)
            score += batch_score.sum()
        score = 100 * score / len(dataloader.dataset)


        return score
