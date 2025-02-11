
import imp
import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime

import test
import util
import commons
import datasets_ws
import network
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.GeoLocalizationNet(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

if args.resume != None:
    state_dict = torch.load(args.resume)["model_state_dict"]
    model.load_state_dict(state_dict)

if args.pca_dim == None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

# default args:
# {'brightness': None,
#  'cache_refresh_rate': 1000,
#  'contrast': None,
#  'criterion': 'triplet',
#  'dataset_name': 'st_lucia',
#  'datasets_folder': './datasets',
#  'dense_feature_map_size': [61, 61, 128],
#  'device': 'cuda',
#  'efficient_ram_testing': False,
#  'epochs_num': 50,
#  'features_dim': 1024,
#  'foundation_model_path': None,
#  'horizontal_flip': False,
#  'hue': None,
#  'infer_batch_size': 16,
#  'l2': 'before_pool',
#  'lr': 1e-05,
#  'majority_weight': 0.01,
#  'margin': 0.1,
#  'mining': 'partial',
#  'neg_samples_num': 1000,
#  'negs_num_per_query': 2,
#  'num_workers': 8,
#  'optim': 'adam',
#  'patience': 3,
#  'pca_dataset_folder': None,
#  'pca_dim': None,
#  'queries_per_epoch': 5000,
#  'rand_perspective': None,
#  'random_resized_crop': None,
#  'random_rotation': None,
#  'recall_values': [1, 5, 10, 20],
#  'registers': True,
#  'rerank_num': 100,
#  'resize': [224, 224],
#  'resume': 'ckpts/SelaVPR_reg4_msls.pth',
#  'saturation': None,
#  'save_dir': 'test/default/2025-02-06_13-22-01',
#  'seed': 0,
#  'test_method': 'hard_resize',
#  'train_batch_size': 4,
#  'train_positives_dist_threshold': 10,
#  'val_positive_dist_threshold': 25}