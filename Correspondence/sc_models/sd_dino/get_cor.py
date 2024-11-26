import os
import torch
from tqdm import tqdm
from PIL import Image
from extractor_sd import process_features_and_mask
from sc_models.sd_dino.cor_utils import resize, pairwise_sim, co_pca, chunk_cosine_sim
import numpy as np
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

MASK = False
SAMPLE = 20
TOTAL_SAVE_RESULT = 5
BBOX_THRE = True
VER = 'v1-5'
CO_PCA = True
CO_PCA_DINO = False
PCA_DIMS = [256, 256, 256]
SIZE = 960
EDGE_PAD = False

FUSE_DINO = True
ONLY_DINO = False
DINOV2 = True
MODEL_SIZE = 'base'
TEXT_INPUT = False

SEED = 42
WEIGHT = [1, 1, 1, 1, 1] # corresponde to three groups for the sd features, and one group for the dino features
PASCAL = False
RAW = False

@torch.no_grad()
def get_cor_pairs(model, aug, extractor, src_image, trg_image, src_points, src_prompt, trg_prompt, dist='l2', transpose_img_func=lambda x:x, transpose_pts_func = lambda x, y: (x, y), device='cuda'):
    sd_size = 960
    dino_size = 840 if DINOV2 else 224 if ONLY_DINO else 480
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
    
    model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if DINOV2 else 'key'
    stride = 14 if DINOV2 else 4 if ONLY_DINO else 8
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (dino_size // patch_size - 1) + 1)

    # Load src image
    src_image = transpose_img_func(Image.open(src_image).convert('RGB'))
    src_w, src_h = src_image.size 
    src_sd_input = resize(src_image, sd_size, resize=True, to_pil=True, edge=EDGE_PAD)
    src_dino_input = resize(src_image, dino_size, resize=True, to_pil=True, edge=EDGE_PAD)
    src_points = [transpose_pts_func(x, y) for x, y in src_points]
    src_x_scale, src_y_scale = dino_size / src_w, dino_size / src_h
    src_points = torch.Tensor([[int(np.round(x * src_x_scale)), int(np.round(y * src_y_scale))] for (x, y) in src_points])
    # Get patch index for the keypoints
    src_y, src_x = src_points[:, 1].numpy(), src_points[:, 0].numpy()
    src_y_patch = (num_patches / dino_size * src_y).astype(np.int32)
    src_x_patch = (num_patches / dino_size * src_x).astype(np.int32)
    src_patch_idx = num_patches * src_y_patch + src_x_patch

    # Load trg image
    trg_image = Image.open(trg_image).convert('RGB')
    trg_w, trg_h = trg_image.size
    trg_sd_input = resize(trg_image, sd_size, resize=True, to_pil=True, edge=EDGE_PAD)
    trg_dino_input = resize(trg_image, dino_size, resize=True, to_pil=True, edge=EDGE_PAD)
    trg_x_scale, trg_y_scale = dino_size / trg_w, dino_size / trg_h
    
    if not CO_PCA:
        if not ONLY_DINO:
            src_desc = process_features_and_mask(model, aug, src_sd_input, input_text=src_prompt, mask=False).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
            trg_desc = process_features_and_mask(model, aug, trg_sd_input, input_text=trg_prompt, mask=False).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
        if FUSE_DINO:
            src_batch = extractor.preprocess_pil(src_dino_input)
            src_desc_dino = extractor.extract_descriptors(src_batch.to(device), layer, facet)
            trg_batch = extractor.preprocess_pil(trg_dino_input)
            trg_desc_dino = extractor.extract_descriptors(trg_batch.to(device), layer, facet)

    else:
        if not ONLY_DINO:
            features1 = process_features_and_mask(model, aug, src_sd_input, input_text=src_prompt,  mask=False, raw=True)
            features2 = process_features_and_mask(model, aug, trg_sd_input, input_text=trg_prompt,  mask=False, raw=True)
            if not RAW:
                processed_features1, processed_features2 = co_pca(features1, features2, PCA_DIMS)
            else:
                if WEIGHT[0]:
                    processed_features1 = features1['s5']
                    processed_features2 = features2['s5']
                elif WEIGHT[1]:
                    processed_features1 = features1['s4']
                    processed_features2 = features2['s4']
                elif WEIGHT[2]:
                    processed_features1 = features1['s3']
                    processed_features2 = features2['s3']
                elif WEIGHT[3]:
                    processed_features1 = features1['s2']
                    processed_features2 = features2['s2']
                else:
                    raise NotImplementedError
                # rescale the features
                processed_features1 = F.interpolate(processed_features1, size=(num_patches, num_patches), mode='bilinear', align_corners=False)
                processed_features2 = F.interpolate(processed_features2, size=(num_patches, num_patches), mode='bilinear', align_corners=False)

            src_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
            trg_desc = processed_features2.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
        if FUSE_DINO:
            src_batch = extractor.preprocess_pil(src_dino_input)
            src_desc_dino = extractor.extract_descriptors(src_batch.to(device), layer, facet)
            trg_batch = extractor.preprocess_pil(trg_dino_input)
            trg_desc_dino = extractor.extract_descriptors(trg_batch.to(device), layer, facet)
    
    if CO_PCA_DINO:
        cat_desc_dino = torch.cat((src_desc_dino, trg_desc_dino), dim=2).squeeze() # (1, 1, num_patches**2, dim)
        mean = torch.mean(cat_desc_dino, dim=0, keepdim=True)
        centered_features = cat_desc_dino - mean
        U, S, V = torch.pca_lowrank(centered_features, q=CO_PCA_DINO)
        reduced_features = torch.matmul(centered_features, V[:, :CO_PCA_DINO]) # (t_x+t_y)x(d)
        processed_co_features = reduced_features.unsqueeze(0).unsqueeze(0)
        src_desc_dino = processed_co_features[:, :, :src_desc_dino.shape[2], :]
        trg_desc_dino = processed_co_features[:, :, src_desc_dino.shape[2]:, :]

    if not ONLY_DINO and not RAW: # reweight different layers of sd

        src_desc[...,:PCA_DIMS[0]]*=WEIGHT[0]
        src_desc[...,PCA_DIMS[0]:PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[1]
        src_desc[...,PCA_DIMS[1]+PCA_DIMS[0]:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[2]

        trg_desc[...,:PCA_DIMS[0]]*=WEIGHT[0]
        trg_desc[...,PCA_DIMS[0]:PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[1]
        trg_desc[...,PCA_DIMS[1]+PCA_DIMS[0]:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[2]

    if 'l1' in dist or 'l2' in dist or dist == 'plus_norm':
        # normalize the features
        src_desc = src_desc / src_desc.norm(dim=-1, keepdim=True)
        trg_desc = trg_desc / trg_desc.norm(dim=-1, keepdim=True)
        src_desc_dino = src_desc_dino / src_desc_dino.norm(dim=-1, keepdim=True)
        trg_desc_dino = trg_desc_dino / trg_desc_dino.norm(dim=-1, keepdim=True)

    if FUSE_DINO and not ONLY_DINO and dist!='plus' and dist!='plus_norm':
        # cat two features together
        src_desc = torch.cat((src_desc, src_desc_dino), dim=-1)
        trg_desc = torch.cat((trg_desc, trg_desc_dino), dim=-1)
        if not RAW:
            # reweight sd and dino
            src_desc[...,:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[3]
            src_desc[...,PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]:]*=WEIGHT[4]
            trg_desc[...,:PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]]*=WEIGHT[3]
            trg_desc[...,PCA_DIMS[2]+PCA_DIMS[1]+PCA_DIMS[0]:]*=WEIGHT[4]

    elif dist=='plus' or dist=='plus_norm':
        src_desc = src_desc + src_desc_dino
        trg_desc = trg_desc + trg_desc_dino
        dist='cos'
    
    if ONLY_DINO:
        src_desc = src_desc_dino
        trg_desc = trg_desc_dino

    # Get similarity matrix
    if dist == 'cos':
        sim_1_to_2 = chunk_cosine_sim(src_desc, trg_desc).squeeze()
    elif dist == 'l2':
        sim_1_to_2 = pairwise_sim(src_desc, trg_desc, p=2).squeeze()
    elif dist == 'l1':
        sim_1_to_2 = pairwise_sim(src_desc, trg_desc, p=1).squeeze()
    elif dist == 'l2_norm':
        sim_1_to_2 = pairwise_sim(src_desc, trg_desc, p=2, normalize=True).squeeze()
    elif dist == 'l1_norm':
        sim_1_to_2 = pairwise_sim(src_desc, trg_desc, p=1, normalize=True).squeeze()
    else:
        raise ValueError('Unknown distance metric')

    # Get nearest neighors
    nn_1_to_2 = torch.argmax(sim_1_to_2[src_patch_idx], dim=1)
    max_sim = torch.max(sim_1_to_2[src_patch_idx], dim=1)[0].detach().cpu().numpy()

    nn_y_patch, nn_x_patch = nn_1_to_2 // num_patches, nn_1_to_2 % num_patches
    nn_x = (nn_x_patch - 1) * stride + stride + patch_size // 2 - .5
    nn_y = (nn_y_patch - 1) * stride + stride + patch_size // 2 - .5
    trg_points = torch.stack([nn_x, nn_y]).permute(1, 0).cpu().numpy()
    trg_points = [[int(np.round(x / trg_x_scale)), int(np.round(y / trg_y_scale))] for (x, y) in trg_points]

    return trg_points, src_points, None, src_dino_input, max_sim

