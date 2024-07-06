
import os
import numpy as np
import pandas as pd
import torch
from bleu import bleu
import _pickle as cPickle

from cfvqa import CFVQA
from confusion_matrix import plot_confusion_matrix


def answer_list(cfg):
    if cfg.DATASET.DATASET == "RAD":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "SLAKE":
        dataroot = r'your_path'

    elif cfg.DATASET.DATASET == "CLEF2020":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "CLEF2019Abnormality":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "CLEF2019Plane":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "CLEF2019Organ":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "CLEF2019Modality":
        dataroot = r'your_path'
    else:
        raise ValueError(f"Dataset {cfg.DATASET.DATASET} is not supported!")
    # bleu
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    ans2label = cPickle.load(open(ans2label_path, 'rb'))  # {ans:idx}
    key_list = list(ans2label.keys())
    return key_list





def compute_score_with_logits(cfg, logits, labels, cnf_matrix):
    if cfg.DATASET.DATASET == "RAD":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "SLAKE":
        dataroot = r'your_path'

    elif cfg.DATASET.DATASET == "CLEF2020":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "CLEF2019Abnormality":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "CLEF2019Plane":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "CLEF2019Organ":
        dataroot = r'your_path'
    elif cfg.DATASET.DATASET == "CLEF2019Modality":
        dataroot = r'your_path'
    else:
        raise ValueError(f"Dataset {cfg.DATASET.DATASET} is not supported!")
    # bleu
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    ans2label = cPickle.load(open(ans2label_path, 'rb'))  # {ans:idx}
    key_list = list(ans2label.keys())
    val_list = list(ans2label.values())
    reference = []
    label = torch.topk(labels, 1)[1].squeeze(1).cpu().numpy()
    for iter in label:
        key_idx = val_list.index(iter)
        reference.append(key_list[key_idx])
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    candidate = []
    one_hot = torch.topk(one_hots, 1)[1].squeeze(1).cpu().numpy()
    for iter in one_hot:
        key_idx = val_list.index(iter)
        candidate.append(key_list[key_idx])
    bleu_score = bleu(candidate, reference, [1])
    scores = (one_hots * labels)


    if cfg.DATASET.DATASET == "CLEF2019Modality":
        list_idx = [0,1,3,8,12,13,14,16,17,21,22,23,25,26,28,30,31,35,36,37,38,39,40,42,45,46]
        list_idx.reverse()
        for i in list_idx:
            key_list.pop(i)
        if reference[0] in key_list and candidate[0] in key_list:
            cnf_matrix[key_list.index(reference[0])][key_list.index(candidate[0])] = \
            cnf_matrix[key_list.index(reference[0])][key_list.index(candidate[0])] + 1


    else:
        if reference[0] in key_list and candidate[0] in key_list:
            cnf_matrix[key_list.index(reference[0])][key_list.index(candidate[0])] = \
            cnf_matrix[key_list.index(reference[0])][key_list.index(candidate[0])] + 1
    cnf_matrix = np.array(cnf_matrix)



    return scores, bleu_score, key_list, cnf_matrix



def save_questiontype_results(outfile_path, quesntion_types_result):
    for i in quesntion_types_result:
        pd.DataFrame(quesntion_types_result[i]).transpose().to_csv(outfile_path + '/question_type_' + i + '.csv')



# Train phase
def test(args, cfg,  model, eval_loader, device,dataset):
    model = model.to(device)

    # Evaluation
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    score, bleu, attack_types, cnf_matrix= evaluate_classifier(args, model, eval_loader, cfg, device, dataset)
    print(score)
    print(bleu)

    plot_confusion_matrix(cnf_matrix, cfg, classes=attack_types,  normalize=False, title='Confusion matrix on %s' )








        
# Evaluation
def evaluate_classifier(args, model, dataloader, cfg, device, dataset):
    score = 0
    bleu = 0
    model.eval()
    cnf_matrix = [[0 for col in range(len(answer_list(cfg)))] for row in range(len(answer_list(cfg)))]
    with torch.no_grad():
        for i, (v, q, a, _, _, _, _, _, _) in enumerate(dataloader):
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
            label = a.to(device)

            if args.fusion_mode == 'rubi' or args.fusion_mode == 'normal':
                q_embedding, _ = model(args, v, q)
                preds = model.classifier(q_embedding)
            elif args.fusion_mode == 'cf_hm' or args.fusion_mode == 'cf_sum' or args.fusion_mode == 'cf_mask':
                cf_vqa = CFVQA(args, cfg, model, args.fusion_mode, dataset)
                cf_vqa_out = cf_vqa(v, q)
                preds = cf_vqa_out['logits_cfvqa']
            else:
                raise ValueError(f"fusion model is not supported!")

            batch_score, bleu_score, attack_types, matrix= compute_score_with_logits(cfg, preds, label.data,cnf_matrix)
            score += batch_score.sum()
            bleu += bleu_score

        score = 100 * score / len(dataloader.dataset)
        bleu = bleu / len(dataloader.dataset)


        return score, bleu, attack_types, matrix


