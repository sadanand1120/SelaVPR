import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import parser
import os
import network
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

###### Modify parameters "match_pattern", "imgpath0" and "imgpath1" according to your needs.
match_pattern = "dense"  # "dense" for matching dense local features (61*61) ; "coarse" for matching coarse patch tokens (16*16)
imgpath0 = "/robodata/smodak/VPR/SelaVPR/datasets/san_francisco/images/test/database/@0550313.87@4184192.09@10@S@037.80373@-122.42845@14666@00@089@003@@@@@.jpg"
imgpath1 = "/robodata/smodak/VPR/SelaVPR/datasets/san_francisco/images/test/database/@0550317.74@4184192.67@10@S@037.80373@-122.42841@14667@00@089@003@@@@@.jpg"

args = parser.parse_arguments()
t = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_patchfeature(model,imgpath):
    img = Image.open(imgpath)
    img = t(img).unsqueeze(0).to(args.device)
    if match_pattern == "dense":
        feature, _ = model(img)
        feature = feature.view(1,61*61,128)
    elif match_pattern == "coarse":  
        feature = model.module.backbone(img)
        feature = feature["x_norm_patchtokens"]
        feature = torch.nn.functional.normalize(feature, p=2, dim=-1)
    return feature

def get_keypoints(img_size): 
    H,W = img_size
    if match_pattern == "dense":
        patch_size = 224/61
    elif match_pattern == "coarse":
        patch_size = 14
    N_h = int(H/patch_size)
    N_w = int(W/patch_size)
    keypoints = np.zeros((2, N_h*N_w), dtype=int) #(x,y)
    keypoints[0] = np.tile(np.linspace(patch_size//2, W-patch_size//2, N_w, 
                                       dtype=int), N_h)
    keypoints[1] = np.repeat(np.linspace(patch_size//2, H-patch_size//2, N_h,
                                         dtype=int), N_w)
    return np.transpose(keypoints)

def match_batch_tensor(fm1, fm2, img_size):
    '''
    fm1: (l,D)
    fm2: (N,l,D)
    mask1: (l)
    mask2: (N,l)
    '''
    M = torch.matmul(fm2, fm1.T) # (N,l,l)
    
    max1 = torch.argmax(M, dim=1) #(N,l)
    max2 = torch.argmax(M, dim=2) #(N,l)
    m = max2[torch.arange(M.shape[0]).reshape((-1,1)), max1] #(N, l)
    valid = torch.arange(M.shape[-1]).repeat((M.shape[0],1)).cuda() == m #(N, l) bool

    kps = get_keypoints(img_size)
    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i,:]).squeeze()
        idx2 = max1[i,:][idx1]
        assert idx1.shape==idx2.shape

        # ############## Filter the nearest neighbor matches by homography verification ###############
        # ### This is not necessary for VPR and not used in SelaVPR. You can comment these four lines of code
        # thetaGT, mask = cv2.findFundamentalMat(kps[idx1.cpu().numpy()],kps[idx2.cpu().numpy()], cv2.FM_RANSAC,
        #                                 ransacReprojThreshold=5)
        # idx1 = idx1[np.where(mask==1)[0]]
        # idx2 = idx2[np.where(mask==1)[0]] 
        # ##############       

        cv_im_one = cv2.resize(cv2.imread(imgpath0),(224,224))
        cv_im_two = cv2.resize(cv2.imread(imgpath1),(224,224))

        kps = get_keypoints(img_size)
        inlier_keypoints_one = kps[idx1.cpu().numpy()]
        inlier_keypoints_two = kps[idx2.cpu().numpy()]
        kp_all1 = []
        kp_all2 = []
        matches_all = []
        print("Number of matched point pairs:", len(inlier_keypoints_one))
        #for this_inlier_keypoints_one, this_inlier_keypoints_two in zip(inlier_keypoints_one, inlier_keypoints_two):
        for k in range(inlier_keypoints_one.shape[0]):
            kp_all1.append(cv2.KeyPoint(inlier_keypoints_one[k, 0].astype(float), inlier_keypoints_one[k, 1].astype(float), 1, -1, 0, 0, -1))
            kp_all2.append(cv2.KeyPoint(inlier_keypoints_two[k, 0].astype(float), inlier_keypoints_two[k, 1].astype(float), 1, -1, 0, 0, -1))
            matches_all.append(cv2.DMatch(k, k, 0))

        im_allpatch_matches = cv2.drawMatches(cv_im_one, kp_all1, cv_im_two, kp_all2,
                                            matches_all, None, matchColor=(0, 255, 0), flags=2)
        cv2.imwrite("patch_matches.jpg",im_allpatch_matches)

model = network.GeoLocalizationNet(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)
state_dict = torch.load(args.resume)["model_state_dict"]
model.load_state_dict(state_dict)

patch_feature0 = get_patchfeature(model,imgpath0)
patch_feature1 = get_patchfeature(model,imgpath1)

def visualize_patch_similarity(p0, p1):
    """
    Computes the similarity between corresponding patches of two images using dot product and visualizes it.
    Args:
        p0: np.array of shape (16, 16, 1024) - First image patch embeddings (normalized).
        p1: np.array of shape (16, 16, 1024) - Second image patch embeddings (normalized).
    """
    p0 = p0.reshape(16,16,1024)
    p1 = p1.reshape(16,16,1024)
    # Convert to NumPy if tensors
    if isinstance(p0, torch.Tensor):
        p0 = p0.detach().cpu().numpy()
    if isinstance(p1, torch.Tensor):
        p1 = p1.detach().cpu().numpy()
    # Compute similarity via dot product (cosine similarity since vectors are normalized)
    similarity_map = np.sum(p0 * p1, axis=-1)  # Shape: (16, 16)

    # Plot similarity heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(similarity_map, cmap="gray", interpolation="nearest")
    plt.colorbar(label="Similarity Score")
    plt.title("Patch-wise Similarity Heatmap")
    plt.axis("off")
    plt.show()

# patch_feature0 = patch_feature0.reshape(61,61,128)
# norm = torch.norm(patch_feature0, dim=-1)  # Shape: (16, 16)
# print("Patch feature 0 shape:",patch_feature0.shape)
# print(norm.shape)
# print(norm[0:5,0:5])
# import ipdb; ipdb.set_trace()
visualize_patch_similarity(patch_feature0, patch_feature1)
import sys
sys.exit()

print("Size of patch tokens:",patch_feature1.shape[1:])
match_batch_tensor(patch_feature0[0], patch_feature1, img_size=(224,224))
