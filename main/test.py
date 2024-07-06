import json
import os
import time
import pandas as pd
import torch
from bleu import bleu
import _pickle as cPickle

from cfvqa import CFVQA

answer_types = ['CLOSED', 'OPEN', 'ALL']
rad_quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']
slake_quesntion_types = ['Plane', 'Quantity', 'Modality', 'Position', 'Organ', 'KG', 'Abnormality', 'Color', 'Size', 'Shape']
clef_quesntion_types = ['is','does','what','are','which','in']

def compute_score_with_logits(cfg, logits, labels):
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
    return scores, bleu_score

def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp

def save_questiontype_results(outfile_path, quesntion_types_result):
    for i in quesntion_types_result:
        pd.DataFrame(quesntion_types_result[i]).transpose().to_csv(outfile_path + '/question_type_' + i + '.csv')



# Train phase
def test(args, cfg,  model, eval_loader, device,dataset):
    model = model.to(device)

    # Evaluation
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    result, question_types_result = evaluate_classifier(args, model, eval_loader, cfg, device, dataset)
    outfile_path = os.path.join(cfg.OUTPUT_DIR, args.fusion_mode)
    outfile = outfile_path + '/results.json'
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    print(result)
    print(question_types_result)
    json.dump(result, open(outfile, 'w'))
    save_questiontype_results(outfile_path, question_types_result)


        
# Evaluation
def evaluate_classifier(args, model, dataloader, cfg, device, dataset):
    keys = ['count', 'real', 'true', 'real_percent', 'score', 'score_percent', 'bleu_score']
    if cfg.DATASET.DATASET == 'RAD':
        question_types_result = dict(
            (i, dict((j, dict((k, 0.0) for k in keys)) for j in rad_quesntion_types)) for i in answer_types)
    elif cfg.DATASET.DATASET == 'SLAKE':
        question_types_result = dict(
            (i, dict((j, dict((k, 0.0) for k in keys)) for j in slake_quesntion_types)) for i in answer_types)
    else:
        question_types_result = dict(
            (i, dict((j, dict((k, 0.0) for k in keys)) for j in clef_quesntion_types)) for i in answer_types)
    result = dict((i, dict((j, 0.0) for j in keys)) for i in answer_types)

    model.eval()
    with torch.no_grad():
        for i,(v, q, a,answer_type, question_type, phrase_type, image_name, question_text, answer_text) in enumerate(dataloader):

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
            a = a.to(device)


            cf_vqa = CFVQA(args, cfg, model, args.fusion_mode, dataset)
            cf_vqa_out = cf_vqa(v, q)
            if args.fusion_mode == 'normal' or args.fusion_mode == 'rubi':
                batch_score, batch_bleu_score = compute_score_with_logits(cfg, cf_vqa_out['logits_vq'], a.data)
            elif args.fusion_mode == 'cf_hm' or args.fusion_mode == 'cf_sum' or args.fusion_mode == 'cf_mask':
                batch_score, batch_bleu_score = compute_score_with_logits(cfg, cf_vqa_out['logits_cfvqa'], a.data)
            else:
                raise ValueError(f"fusion_mode is not supported!")



            batch_score = batch_score.sum()

            # Compute accuracy for each type answer
            result[answer_type[0]]['count'] += 1.0
            result[answer_type[0]]['true'] += float(batch_score)
            result[answer_type[0]]['real'] += float(a.sum())
            result[answer_type[0]]['bleu_score'] += float(batch_bleu_score)

            result['ALL']['count'] += 1.0
            result['ALL']['true'] += float(batch_score)
            result['ALL']['real'] += float(a.sum())

            question_type = question_type[0].split(", ")
            for i in question_type:
                question_types_result[answer_type[0]][i]['count'] += 1.0
                question_types_result[answer_type[0]][i]['true'] += float(batch_score)
                question_types_result[answer_type[0]][i]['real'] += float(a.sum())

                question_types_result['ALL'][i]['count'] += 1.0
                question_types_result['ALL'][i]['true'] += float(batch_score)
                question_types_result['ALL'][i]['real'] += float(a.sum())

        for i in answer_types:
            result[i]['score'] = result[i]['true'] / result[i]['real']
            if i == 'ALL':
                result[i]['bleu_score'] = result['OPEN']['bleu_score'] + result['CLOSED']['bleu_score']
            else:
                result[i]['bleu_score'] = result[i]['bleu_score'] / len(dataloader)
            result[i]['score_percent'] = round(result[i]['score'] * 100, 1)

            if cfg.DATASET.DATASET == 'RAD':
                quesntion_types = rad_quesntion_types
            elif cfg.DATASET.DATASET == 'SLAKE':
                quesntion_types = slake_quesntion_types
            else:
                quesntion_types = clef_quesntion_types
            for j in quesntion_types:
                if question_types_result[i][j]['count'] != 0.0:
                    question_types_result[i][j]['score'] = question_types_result[i][j]['true'] / question_types_result[i][j]['real']
                    question_types_result[i][j]['score_percent'] = round(question_types_result[i][j]['score'] * 100, 1)
                if question_types_result[i][j]['real'] != 0.0:
                    question_types_result[i][j]['real_percent'] = round(
                        question_types_result[i][j]['real'] / question_types_result[i][j]['real'] * 100.0, 1)
        return result, question_types_result


