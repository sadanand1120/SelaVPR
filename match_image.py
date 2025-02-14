import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import parser
import os
import network
from tqdm import tqdm
from local_matching import local_sim
import faiss
import warnings
from easydict import EasyDict as edict
import torch.nn.functional as F

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class SelaVPRminimal:
    def __init__(self):
        self.t = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.args = self.get_default_args()
        with torch.device(self.args.device):
            self.model = network.GeoLocalizationNet(self.args)
            self.model = self.model.to(self.args.device)
            self.model = torch.nn.DataParallel(self.model)
            state_dict = torch.load(self.args.resume, map_location=self.args.device)["model_state_dict"]
            self.model.load_state_dict(state_dict)
            self.model.eval()

    @torch.inference_mode()
    def run_model(self, imgpaths: list):
        with torch.device(self.args.device):
            if type(imgpaths[0]) == str:
                images = [Image.open(imgpath).convert('RGB') for imgpath in imgpaths]
            else:
                images = imgpaths
            batch_imgs = torch.stack([self.t(img) for img in images])
            local_feature, global_feature, patch_feature = self.model(batch_imgs)
        return local_feature, global_feature, patch_feature

    def get_default_args(self):
        return edict({'brightness': None,
                      'cache_refresh_rate': 1000,
                      'contrast': None,
                      'criterion': 'triplet',
                      'dataset_name': 'st_lucia',
                      'datasets_folder': './datasets',
                      'dense_feature_map_size': [61, 61, 128],
                      'device': 'cpu',
                      'efficient_ram_testing': False,
                      'epochs_num': 50,
                      'features_dim': 1024,
                      'foundation_model_path': None,
                      'horizontal_flip': False,
                      'hue': None,
                      'infer_batch_size': 16,
                      'l2': 'before_pool',
                      'lr': 1e-05,
                      'majority_weight': 0.01,
                      'margin': 0.1,
                      'mining': 'partial',
                      'neg_samples_num': 1000,
                      'negs_num_per_query': 2,
                      'num_workers': 8,
                      'optim': 'adam',
                      'patience': 3,
                      'pca_dataset_folder': None,
                      'pca_dim': None,
                      'queries_per_epoch': 5000,
                      'rand_perspective': None,
                      'random_resized_crop': None,
                      'random_rotation': None,
                      'recall_values': [1, 5, 10, 20],
                      'registers': False,
                      'rerank_num': 100,
                      'resize': [224, 224],
                      'resume': 'ckpts/SelaVPR_msls.pth',
                      'saturation': None,
                      'save_dir': 'test/default/2025-02-06_13-22-01',
                      'seed': 0,
                      'test_method': 'hard_resize',
                      'train_batch_size': 4,
                      'train_positives_dist_threshold': 10,
                      'val_positive_dist_threshold': 25})

    def show_attnmap(self, imgpath):
        img = Image.open(imgpath).convert('RGB')
        W, H = img.size
        local_feature, global_feature, patch_feature = self.run_model([imgpath])
        fm2 = patch_feature.squeeze(0).reshape(16, 16, 1024)
        A = fm2.mean(dim=-1)
        A_resized = F.interpolate(A.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
        A_resized = A_resized.squeeze().cpu().detach().numpy()
        A_resized = np.abs(A_resized)

        fig, ax = plt.subplots()
        ax.imshow(img)  # Display original image
        im = ax.imshow(A_resized, cmap='jet', alpha=0.5)  # Overlay heatmap with transparency
        ax.axis('off')
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)  # Adjust colorbar label size for readability
        plt.show()

    def match_global_feature(self, query_feat, candidates_feat, K=100):
        query_feat = query_feat.cpu().detach().numpy().reshape(-1, 1024)
        candidates_feat = candidates_feat.cpu().detach().numpy().reshape(-1, 1024)
        faiss_index = faiss.IndexFlatL2(1024)
        faiss_index.add(candidates_feat)
        distances, indices = faiss_index.search(query_feat, K)
        return distances, indices

    def match_local_feature(self, query_feat, candidates_feat, indices, retscores=False):
        query_feat = query_feat.cpu().detach().numpy().reshape(-1, 61, 61, 128)
        candidates_feat = candidates_feat.cpu().detach().numpy().reshape(-1, 61, 61, 128)
        return self.rerank(indices, query_feat, candidates_feat, retscores=retscores)

    @torch.inference_mode()
    def rerank(self, predictions, queries_local_features, database_local_features, retscores=False):
        with torch.device(self.args.device):
            pred2 = []
            highest_scores = []
            print("reranking...")
            for query_index, pred in enumerate(tqdm(predictions)):
                query_local_features = queries_local_features[query_index]
                candidates_local_features = database_local_features[pred]
                query_local_features = torch.Tensor(query_local_features)
                candidates_local_features = torch.Tensor(candidates_local_features)
                scores = local_sim(query_local_features, candidates_local_features).cpu().numpy()
                highest_scores.append(scores.max())
                rerank_index = scores.argsort()[::-1]
                pred2.append(predictions[query_index][rerank_index])
        if retscores:
            return np.array(pred2), highest_scores
        else:
            return np.array(pred2)

    def match_image(self, query_imgpath, candidates_imgpaths, retscores=False):
        all_imgpaths = [query_imgpath] + candidates_imgpaths
        local_feats, global_feats, patch_feats = self.run_model(all_imgpaths)
        query_local_feat, query_global_feat = local_feats[0].squeeze().unsqueeze(0), global_feats[0].squeeze().unsqueeze(0)
        candidates_local_feats, candidates_global_feats = local_feats[1:], global_feats[1:]
        K = 10 if len(candidates_imgpaths) > 10 else len(candidates_imgpaths)
        distances, indices = self.match_global_feature(query_global_feat, candidates_global_feats, K=K)
        return self.match_local_feature(query_local_feat, candidates_local_feats, indices, retscores=retscores)


if __name__ == "__main__":
    sela = SelaVPRminimal()
    imgpath1 = "spot_ex1.png"
    imgpath2 = "spot_ex2.png"
    imgpath3 = "spot_ex3.png"
    imgpath4 = "spot_ex4.png"
    local_feats, global_feats, patch_feats = sela.run_model([imgpath1, imgpath2, imgpath3, imgpath4])
