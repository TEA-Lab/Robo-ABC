import os
import re
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import logging
import argparse
import shutil

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
            
def get_cor_cfg(method):
    cor_cfg = {}
    if method == 'dift':
        cor_cfg['img_size'] = 768
        cor_cfg['ensemble_size'] = 8
    elif method == 'ldm_sc':
        cor_cfg['img_size'] = 512
    elif method == 'sd_dino':
        cor_cfg['model_type'] = 'dinov2_vitb14'
    elif method == 'dino_vit':
        cor_cfg['img_size'] = 256
        cor_cfg['model_type'] = 'dino_vits8'
        cor_cfg['stride'] = 4
    return cor_cfg

def get_cor_pairs(method, model, src_image, trg_image, src_points, src_prompt, trg_prompt, cfg, transpose_img_func=lambda x:x, transpose_pts_func=lambda x, y: (x, y), device='cuda'):
    if method == 'dift':
        from sc_models.dift.get_cor import get_cor_pairs
        return get_cor_pairs(model, src_image, trg_image, src_points, src_prompt, trg_prompt, cfg['img_size'], cfg['ensemble_size'], return_cos_maps=cfg['visualize'], transpose_img_func=transpose_img_func, transpose_pts_func=transpose_pts_func)
    elif method == 'ldm_sc': # ldm_sc don't get transpose_img_func and transpose_pts_func cause it will be too slow.
        from sc_models.ldm_sc.get_cor import get_cor_pairs
        return get_cor_pairs(model, src_image, trg_image, src_points, cfg['img_size'], device), None
    elif method == 'sd_dino':
        from sc_models.sd_dino.get_cor import get_cor_pairs
        model, aug, extractor = model
        return get_cor_pairs(model, aug, extractor, src_image, trg_image, src_points, src_prompt, trg_prompt, transpose_img_func=transpose_img_func, transpose_pts_func=transpose_pts_func, device=device)
    elif method == 'dino_vit':
        from sc_models.dino_vit.get_cor import get_cor_pairs
        return get_cor_pairs(model, src_image, trg_image, src_points, cfg['img_size'], transpose_img_func=transpose_img_func, transpose_pts_func=transpose_pts_func, device=device)
    else:
        raise NotImplementedError

def get_model(method, cor_cfg, device='cuda'):
    if method == 'dift':
        from sc_models.dift.dift_sd import SDFeaturizer
        return SDFeaturizer(device)
    elif method == 'ldm_sc':
        from sc_models.ldm_sc.optimize import load_ldm
        return load_ldm(device, 'CompVis/stable-diffusion-v1-4')
    elif method == 'sd_dino':
        from sc_models.sd_dino.extractor_sd import load_model
        from sc_models.sd_dino.extractor_dino import ViTExtractor
        model_type = cor_cfg['model_type']
        stride = 14 if 'v2' in model_type else 8
        extractor = ViTExtractor(model_type, stride, device=device)
        model, aug = load_model(diffusion_ver='v1-5', image_size=960, num_timesteps=100, block_indices=(2,5,8,11))
        return model, aug, extractor
    elif method == 'dino_vit':
        from sc_models.dino_vit.extractor import ViTExtractor
        model_type = cor_cfg['model_type']
        stride = cor_cfg['stride']
        return ViTExtractor(model_type, stride, device=device)

def plot_img_pairs(imglist, src_points_list, trg_points_list, trg_mask, top_k=1, cos_map_list=None, save_name='corr.png', fig_size=5, alpha=0.45, scatter_size=30):
    num_imgs = top_k + 1
    src_images = len(imglist) - 1
    fig, axes = plt.subplots(src_images, num_imgs + 1, figsize=(fig_size*(num_imgs + 1), fig_size*src_images))
    plt.tight_layout()

    for i in range(src_images):
        ax = axes[i] if src_images > 1 else axes
        ax[0].imshow(imglist[i])
        ax[0].axis('off')
        ax[0].set_title('source')
        for x, y in src_points_list[i]:
            x, y = int(np.round(x)), int(np.round(y))
            ax[0].scatter(x, y, s=scatter_size)

        for j in range(1, num_imgs):
            ax[j].imshow(imglist[-1])
            if cos_map_list[0] is not None:
                heatmap = cos_map_list[i][j - 1][0]
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                ax[j].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
            ax[j].axis('off')
            ax[j].scatter(trg_points_list[i][j - 1][0], trg_points_list[i][j - 1][1], c='C%d' % (j - 1), s=scatter_size)
            ax[j].set_title('target')
        
        ax[-1].imshow(trg_mask, cmap='gray')
        ax[-1].axis('off')
        ax[-1].set_title('target mask')
        trg_point = np.mean(trg_points_list[i], axis=0)
        ax[-1].scatter(trg_point[0], trg_point[1], c='C%d' % (j - 1), s=scatter_size)
    plt.plot()
    plt.savefig(save_name)
    plt.close()


def nearest_distance_to_mask_contour(mask, x, y, threshold=122, stride=30):
    # Convert the boolean mask to an 8-bit image
    dist_list = []
    last_mask = ((mask > 0).astype(np.uint8) * 255)
    threshold_list = list(range(0, 255, stride)) + [threshold] # the last one is the threshold value
    for mask_threshold in threshold_list:
        mask_8bit = ((mask > mask_threshold).astype(np.uint8) * 255)
        if mask_8bit.sum() == 0:
            mask_8bit = (mask == mask.max()).astype(np.uint8) * 255
        # Find the contours in the mask
        contours, _ = cv2.findContours(mask_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Check if point is inside any contour
        num = 0
        y = min(y, np.array(mask).shape[0] - 1)
        x = min(x, np.array(mask).shape[1] - 1)
        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) == 1:  # Inside contour
                num += 1
        if num % 2 == 1:
            dist_list.append(0)
            continue
        
        # If point is outside all contours, find the minimum distance between the point and each contour
        min_distance = float('inf')
        for contour in contours:
            distance = cv2.pointPolygonTest(contour, (x, y), True)  # Measure distance
            if abs(distance) < min_distance:
                min_distance = abs(distance)
        
        
        # normalize the distance with the diagonal length of the mask
        diag_len = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
        dist_list.append(abs(min_distance) / diag_len)
    nss_value = np.array(mask)[int(y), int(x)]
    thres_dist = dist_list.pop()
    return dist_list, nss_value, thres_dist


def dataset_walkthrough(base_dir, method, model, exp_name, cor_cfg={}, average_pts=True, visualize=False, mask_threshold=120, top_k=1, top_k_type='max', transpose_types=1, device='cuda'):
    eval_pairs = 0
    total_dists, nss_values, thres_dists, res_trg_points = {}, {}, {}, {}
    gt_dir = os.path.join(base_dir, 'GT')
    base_dir = os.path.join(base_dir, 'egocentric')
    transpose_img_funcs = [
        lambda x:x,
        lambda x:x.rotate(90, expand=True),
        lambda x:x.rotate(180, expand=True),
        lambda x:x.rotate(-90, expand=True),
        lambda x:x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x:x.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True),
        lambda x:x.transpose(Image.FLIP_LEFT_RIGHT).rotate(180, expand=True),
        lambda x:x.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True),
    ]
    for trg_object in os.listdir(base_dir):
        eval_pairs += len(os.listdir(os.path.join(base_dir, trg_object)))
    print(f'Start evaluating {eval_pairs} correspondance pairs...')
    
    cor_cfg['device'] = device
    cor_cfg['visualize'] = visualize

    pbar = tqdm(total=eval_pairs)
    if visualize:
        if os.path.exists(f'results_arxiv/{method}/{exp_name}'):
            confrim = input(f'The result folder {method}/{exp_name} already exists. input y to remove it...')
            if confrim == 'y':
                shutil.rmtree(f'results_arxiv/{method}/{exp_name}', ignore_errors=True)
            else:
                exit()
    for trg_object in os.listdir(base_dir):
        object_path = os.path.join(base_dir, trg_object)
        total_dists[trg_object], nss_values[trg_object], thres_dists[trg_object], res_trg_points[trg_object] = [], [], [], {}
        for instance in os.listdir(object_path):
            instance_path = os.path.join(object_path, instance)
            src_images, src_points_list, trg_points_list, cor_map_list, src_object_list = [], [], [], [], []
            for file in os.listdir(instance_path):
                if file.endswith('.jpg'):
                    trg_image = os.path.join(instance_path, file)
                    mask_file = os.path.join(gt_dir, trg_object, file.replace('jpg', 'png'))
                    with Image.open(mask_file) as img:
                        try:
                            trg_mask = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                        except:
                            trg_mask = np.array(img)
                elif file.endswith('.txt') and ('top' not in file or int(file.strip('.txt').split('top')[1]) <= top_k):
                    src_images.append(os.path.join(instance_path, file).replace('txt', 'png'))
                    src_object_list.append(file.split('_')[0])
                    with open(os.path.join(instance_path, file), 'r') as f:
                        lines = f.readlines()
                        src_points = [list(map(float, line.rstrip().split(','))) for line in lines if re.match(r'^\d+.\d+,.*\d+.\d+$', line.rstrip())]
                        if average_pts:
                            src_points = [np.mean(np.array(src_points), axis=0).astype(np.int32)]
                        src_points_list.append(src_points)
            pbar.set_description(f'{trg_object}-{instance}')
            trg_prompt = f'a photo of {trg_object}'
            imglist, new_src_points_list, cor_values_list = [], [], []
            for i in range(len(src_images)):
                src_prompt = f'a photo of a {src_object_list[i]}'
                w, h = Image.open(src_images[i]).size
                transpose_pts_funcs = [
                    lambda x, y: (x, y),
                    lambda x, y: (y, w - x), # 90 
                    lambda x, y: (w - x, h - y), # 180
                    lambda x, y: (h - y, x), # -90
                    lambda x, y: (w - x, y), # flip
                    lambda x, y: (y, x), # flip 90
                    lambda x, y: (x, h - y), # flip 180
                    lambda x, y: (h - y, w - x), # flip -90
                ]
                trg_pnts_tmp, src_pnts_tmp, cor_maps_tmp, src_image_tmp, cor_values_tmp = [], [], [], [], []
                os.makedirs(f'results_arxiv/{method}/{exp_name}/{trg_object}', exist_ok=True)
                for j in range(transpose_types):
                    trg_pnts, src_pnts, cor_maps, src_image, cor_values = get_cor_pairs(method, model, src_images[i], trg_image, src_points_list[i], src_prompt, trg_prompt, cor_cfg, transpose_img_funcs[j], transpose_pts_funcs[j], device)
                    trg_pnts_tmp.append(trg_pnts)
                    src_pnts_tmp.append(src_pnts)
                    cor_maps_tmp.append(cor_maps)
                    src_image_tmp.append(src_image)
                    cor_values_tmp.append(np.mean(cor_values))
                    # cor_values_str = ", ".join(map(lambda x: "%.2f" % x, cor_values))
                    # plot_img_pairs([src_image, Image.open(trg_image).convert('RGB')], [src_pnts], [trg_pnts], trg_mask, [cor_maps], os.path.join(f'results_arxiv/{method}/{exp_name}/{trg_object}', f'{instance}_{j}_{cor_values_str}.png'))
                selected_idx = np.argmax(cor_values_tmp)
                new_src_points_list.append(src_pnts_tmp[selected_idx])
                trg_points_list.append(trg_pnts_tmp[selected_idx])
                cor_map_list.append(cor_maps_tmp[selected_idx])
                cor_values_list.append(cor_values_tmp[selected_idx])
                imglist.append(src_image_tmp[selected_idx])
            trg_points = np.mean(trg_points_list, axis=1)
            if top_k_type == 'max':
                trg_point = trg_points[np.argmax(cor_values_list)]
            elif top_k_type == 'avg':
                trg_point = np.mean(trg_points, axis=0)
            trg_dist, nss_value, thres_dist = nearest_distance_to_mask_contour(trg_mask, trg_point[0], trg_point[1], mask_threshold)
            total_dists[trg_object].append(trg_dist)
            thres_dists[trg_object].append(thres_dist)
            nss_values[trg_object].append(nss_value)
            res_trg_points[trg_object][instance] = trg_points_list
            # print(trg_point, trg_dist)ipy
            if visualize:
                res_dir =  f'results_arxiv/{method}/{exp_name}/{trg_object}'
                imglist.append(Image.open(trg_image).convert('RGB'))
                os.makedirs(res_dir, exist_ok=True)
                file_name = f'{instance}_{thres_dist:.2f}_{nss_value}'
                if top_k_type == 'max':
                    file_name += f'_max_idx{np.argmax(cor_values_list)}'
                plot_img_pairs(imglist, new_src_points_list, trg_points_list, trg_mask, top_k, cor_map_list, os.path.join(res_dir, file_name + '.png'))
            pbar.update(1)
    pbar.close()
    return total_dists, nss_values, thres_dists, res_trg_points

    
def analyze_dists(total_dists, nss_values, thres_dists, res_dir=None):
    all_dists, all_nss, thres_dist, lines = [], [], [], []
    
    if res_dir is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20,5))
        for trg_object in total_dists.keys():
            all_dists += total_dists[trg_object]
            dist_curve = np.array(total_dists[trg_object]).mean(axis=0)
            sr_curve = (np.array(total_dists[trg_object]) == 0).sum(axis=0) / len(total_dists[trg_object])
            axes[0].plot(dist_curve, label=trg_object)
            axes[1].plot(sr_curve, label=trg_object)

        axes[0].plot(np.array(all_dists).mean(axis=0), label='all', linewidth=3, color='black')
        axes[1].plot((np.array(all_dists) == 0).sum(axis=0) / len(all_dists), label='all', linewidth=3, color='black')
        axes[0].legend()
        axes[1].legend()
        axes[0].set_title('DTM')
        axes[1].set_title('SR')

        plt.savefig(os.path.join(res_dir, 'dist_curve.png'))
    
    for trg_object in thres_dists.keys():
        all_nss += nss_values[trg_object]
        thres_dist += thres_dists[trg_object]
        lines.append(f'{trg_object.split("_")[0]:12s}: dist mean:{np.array(thres_dists[trg_object]).mean():.3f}, nss mean: {np.array(nss_values[trg_object]).mean():.1f}, success rate: {(np.array(thres_dists[trg_object]) == 0).sum() / len(thres_dists[trg_object]):.3f} ({(np.array(thres_dists[trg_object]) == 0).sum()}/{len(thres_dists[trg_object])})')
    lines.append(f'=== ALL  ===: dist mean:{np.mean(thres_dist):.3f}, nss mean: {np.array(all_nss).mean():.1f}, success rate: {(np.array(thres_dist)==0).sum() / len(thres_dist):.3f} ({(np.array(thres_dist)==0).sum()}/{len(thres_dist)})')
    if res_dir is not None:
        with open(os.path.join(res_dir, 'total_dists.txt'), 'w') as f:
            f.writelines([line + '\n' for line in lines])
    for line in lines:
        print(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', type=str, default='dift', choices=['dift', 'ldm_sc', 'sd_dino', 'dino_vit'], help='method for correspondance')
    parser.add_argument('--dataset', '-d', type=str, default='clip_b32_x2', choices=['clip_b32', 'clib_b32_x0.5', 'clip_b32_x2', 'clip_b16', 'clip_b16_x0.5', 'clip_b16_x2', 'clip_b32_lpips', 'clip_b32_lpips_x0.5', 'clip_b32_lpips_x2', 'resnet_50', 'resnet_50_x0.5', 'resnet_50_x2'], help='dataset for affordance memory')
    parser.add_argument('--exp_name', '-e', type=str, default='', help='experiment name')
    parser.add_argument('--mask_threshold', '-s', type=int, default=122, help='mask threshold for success rate calculation')
    parser.add_argument('--visualize', '-v', action='store_true', help='visualize the correspondance pairs')
    parser.add_argument('--avg_pts', '-a', action='store_true', help='average the five source points before (True) or after (False) correspondance')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='use the top k retrieved images')
    parser.add_argument('--top_k_type', '-kt', type=str, default='max', choices=['max', 'avg'], help='max: use the top k with max cor value, avg: use the average results of top k')
    parser.add_argument('--transpose_types', '-t', type=int, default=8, help='1: no transpose, 4: rotations, 8: flip and rotations')
    args = parser.parse_args()
    args.avg_pts = False
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    average_pts, visualize = args.avg_pts, args.visualize
    exp_name = args.dataset
    if args.mask_threshold != 122:
        exp_name += '_s' + str(args.mask_threshold)
    if args.top_k != 1:
        exp_name += f'_top{args.top_k}'
        if args.top_k_type == 'max':
            exp_name += '_max'
    if args.avg_pts:
        exp_name += '_avg'
    if args.transpose_types > 1:
        exp_name += '_transpose'
    exp_name = exp_name if len(args.exp_name) == 0 else args.exp_name
    cor_cfg = get_cor_cfg(args.method)

    model = get_model(args.method, cor_cfg, device=device)
    base_dir = f'datasets/{args.dataset}'
    res_dir = f'results/{args.method}/{exp_name}'
    print(f'res_dis: {res_dir}')

    total_dists, nss_values, thres_dists, trg_points = dataset_walkthrough(base_dir, args.method, model, exp_name, cor_cfg, average_pts, visualize, args.mask_threshold, args.top_k, args.top_k_type, args.transpose_types, device)
    
    with open(os.path.join(res_dir, 'results.pkl'), 'wb') as f:
        pickle.dump({'args': args, 'total_dists': total_dists, 'nss_values': nss_values, 'thres_dists': thres_dists, 'trg_points': trg_points}, f)

    analyze_dists(total_dists, nss_values, thres_dists, res_dir)