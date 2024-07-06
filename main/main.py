
import argparse
import os

import torch
from torch.utils.data import DataLoader

import clef2019test
from dataset_CLEF2020 import VQACLEF2020FeatureDataset
from dataset_RAD import VQARADFeatureDataset
from dataset_SLAKE import VQASLAKEFeatureDataset
from multi_level_model import BAN_Model
from lib.init import cfg, update_config

from create_dictionary import Dictionary
from test import test
from train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")
    # cfg
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default='../configs/pubmedclipRN50_clef2019Abnormality.yaml',
            type=str
            )
    parser.add_argument('--output', type=str, default='results',
                        help='output file directory for saving VQA answer prediction file')
    # GPU config
    parser.add_argument('--gpu', type=int, default=0,help='use gpu device')
    parser.add_argument('--test', type=bool, default=False, help='Test or train.')
    parser.add_argument('--fusion_mode', type=str, default='cf_hm', choices=['rubi', 'cf_hm', 'cf_sum','cf_mask','normal'])
    parser.add_argument('--question_loss_weight', type=float, default=1)
    parser.add_argument('--vision_loss_weight', type=float, default=1)

    parser.add_argument('--k', type=float, default=0.06)


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    torch.cuda.empty_cache()

    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    update_config(cfg, args)
    data_dir = cfg.DATASET.DATA_DIR
    args.data_dir = data_dir
    # Fixed random seed
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)



    d = Dictionary.load_from_file(data_dir + '\dictionary.pkl')
    if cfg.DATASET.DATASET == "RAD":
        train_dataset = VQARADFeatureDataset('train', cfg, d, dataroot=r'your_path')
        val_dataset = VQARADFeatureDataset('test', cfg, d, dataroot=r'your_path')
    elif cfg.DATASET.DATASET == "SLAKE":
        train_dataset = VQASLAKEFeatureDataset('train', cfg, d, dataroot=r'your_path')
        val_dataset = VQASLAKEFeatureDataset('test', cfg, d, dataroot=r'your_path')
    elif cfg.DATASET.DATASET == "CLEF2020":
        train_dataset = VQACLEF2020FeatureDataset('train', cfg, d, dataroot=r'your_path')
        val_dataset = VQACLEF2020FeatureDataset('test', cfg, d, dataroot=r'your_path')
    elif cfg.DATASET.DATASET == "CLEF2019Abnormality":
        train_dataset = VQACLEF2020FeatureDataset('train', cfg, d, dataroot=r'your_path')
        val_dataset = VQACLEF2020FeatureDataset('test', cfg, d, dataroot=r'your_path')
    elif cfg.DATASET.DATASET == "CLEF2019Plane":
        train_dataset = VQACLEF2020FeatureDataset('train', cfg, d, dataroot=r'your_path')
        val_dataset = VQACLEF2020FeatureDataset('test', cfg, d, dataroot=r'your_path')
    elif cfg.DATASET.DATASET == "CLEF2019Organ":
        train_dataset = VQACLEF2020FeatureDataset('train', cfg, d, dataroot=r'your_path')
        val_dataset = VQACLEF2020FeatureDataset('test', cfg, d, dataroot=r'your_path')
    elif cfg.DATASET.DATASET == "CLEF2019Modality":
        train_dataset = VQACLEF2020FeatureDataset('train', cfg, d, dataroot=r'your_path')
        val_dataset = VQACLEF2020FeatureDataset('test', cfg, d, dataroot=r'your_path')
    else:
        raise ValueError(f"Dataset {cfg.DATASET.DATASET} is not supported!")

    drop_last = False
    drop_last_val = False 
    train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0,drop_last=drop_last,pin_memory=True)
    val_loader = DataLoader(val_dataset, cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=0,drop_last=drop_last_val,pin_memory=True)


    

    if args.test:
        model = BAN_Model(val_dataset, cfg, device)
        model_path = os.path.join(cfg.OUTPUT_DIR, args.fusion_mode)
        model_data = torch.load(model_path + '/best_model.pth')
        model.load_state_dict(model_data.get('model_state', model_data), strict=False)
        if 'CLEF2019' in cfg.DATASET.DATASET :
            clef2019test.test(args, cfg, model, val_loader, device, val_dataset)
        else:
            test(args, cfg,  model, val_loader, device, val_dataset)

    else:
        model = BAN_Model(train_dataset, cfg, device)
        train(args, cfg, model, train_loader, val_loader, device,train_dataset)
        test_model = BAN_Model(val_dataset, cfg, device)
        model_path = os.path.join(cfg.OUTPUT_DIR, args.fusion_mode)
        model_data = torch.load(model_path + '/best_model.pth')
        test_model.load_state_dict(model_data.get('model_state', model_data), strict=False)
        if 'CLEF2019' in cfg.DATASET.DATASET:
            clef2019test.test(args, cfg, model, val_loader, device, val_dataset)
        else:
            test(args, cfg, model, val_loader, device, val_dataset)


