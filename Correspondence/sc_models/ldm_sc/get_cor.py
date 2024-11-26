import re
import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sc_models.ldm_sc.optimize import optimize_prompt, run_image_with_tokens_cropped, find_max_pixel_value, load_ldm


def get_cor_pairs(ldm, src_image, trg_image, src_points, img_size, device='cuda:0'):
    """
    src_image, trg_image: relative path of src and trg images
    src_points: resized affordance points in src_image
    average_pts: average before correspondance or not
    -----
    return: correspondance maps of each src_point and each target_point
    """
    trg_points = []
    layers = [5,6,7,8]
    with Image.open(src_image) as img:
        src_w, src_h = img.size
        src_image = img.resize((img_size, img_size), Image.BILINEAR).convert('RGB')
        src_tensor = torch.Tensor(np.array(src_image).transpose(2, 0, 1)) / 255.0
        src_x_scale, src_y_scale = img_size / src_w, img_size / src_h
    with Image.open(trg_image) as img:
        trg_w, trg_h = img.size
        trg_image = img.resize((img_size, img_size), Image.BILINEAR).convert('RGB')
        trg_tensor = torch.Tensor(np.array(trg_image).transpose(2, 0, 1)) / 255.0
        trg_x_scale, trg_y_scale = img_size / trg_w, img_size / trg_h
    
    src_points = [torch.Tensor([int(np.round(x * src_x_scale)), int(np.round(y * src_y_scale))]) for (x, y) in src_points]
    all_contexts = []
    for src_point in src_points:
        contexts = []
        for _ in range(5):
            context = optimize_prompt(ldm, src_tensor, src_point/img_size, num_steps=129, device=device, layers=layers, lr = 0.0023755632081200314, upsample_res=img_size, noise_level=-8, sigma = 27.97853316316864, flip_prob=0.0, crop_percent=93.16549294381423)
            contexts.append(context)
        all_contexts.append(torch.stack(contexts))
        
        all_maps = []
        for context in contexts:
            maps = []
            attn_maps, _ = run_image_with_tokens_cropped(ldm, trg_tensor, context, index=0, upsample_res = img_size, noise_level=-8, layers=layers, device=device, crop_percent=93.16549294381423, num_iterations=20, image_mask = None)
            for k in range(attn_maps.shape[0]):
                avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
                maps.append(avg)
            maps = torch.stack(maps, dim=0)
            all_maps.append(maps)
        all_maps = torch.stack(all_maps, dim=0)
        all_maps = torch.mean(all_maps, dim=0)
        all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), img_size*img_size))
        all_maps = all_maps.reshape(len(layers), img_size, img_size)
        
        all_maps = torch.mean(all_maps, dim=0)
        trg_points.append(find_max_pixel_value(all_maps, img_size = img_size).cpu().numpy())
    
    
    trg_points = [[int(np.round(x / trg_x_scale)), int(np.round(y / trg_y_scale))] for (x, y) in trg_points]
    
    return trg_points

