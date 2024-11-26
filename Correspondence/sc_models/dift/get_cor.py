import gc
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms import PILToTensor

def pad_image(image, pixel_locs=[]):
    width, height = image.size
    # Calculate padding to make the image square
    if width > height:
        # Width is greater, pad height
        padding = (width - height) // 2
        padded_image = Image.new("RGB", (width, width), (255, 255, 255))
        padded_image.paste(image, (0, padding))
        padded_pixel_locs = [(pixel_loc[0], pixel_loc[1] + padding) for pixel_loc in pixel_locs]
    else:
        # Height is greater, pad width
        padding = (height - width) // 2
        padded_image = Image.new("RGB", (height, height), (255, 255, 255))
        padded_image.paste(image, (padding, 0))
        padded_pixel_locs = [(pixel_loc[0] + padding, pixel_loc[1]) for pixel_loc in pixel_locs]

    return padded_image, padded_pixel_locs

def crop_array(array, org_h, org_w):
    org_h, org_w = round(org_h), round(org_w)
    padding = abs((org_w - org_h) // 2)
    # Convert the new location back to the original image's coordinates
    if org_w > org_h:
        # If the original width was greater, adjust the y-coordinate
        cropped_array = array[:, padding: -padding, :]
    else:
        # If the original height was greater, adjust the x-coordinate
        cropped_array = array[:, :, padding: -padding]
    return cropped_array


def get_cor_pairs(dift, src_image, trg_image, src_points, src_prompt, trg_prompt, img_size, ensemble_size=8, return_cos_maps=False, transpose_img_func=lambda x:x, transpose_pts_func = lambda x, y: (x, y)):
    """
    src_image, trg_image: relative path of src and trg images
    src_points: resized affordance points in src_image
    average_pts: average before correspondance or not
    -----
    return: correspondance maps of each src_point and each target_point
    """
    trg_points = []

    with Image.open(src_image) as img:
        # src_image, src_points = pad_image(img, src_points)
        src_image = transpose_img_func(img)
        src_w, src_h = src_image.size 
        src_image = src_image.resize((img_size, img_size)).convert('RGB')
        src_points = [transpose_pts_func(x, y) for x, y in src_points]
        src_x_scale, src_y_scale = img_size / src_w, img_size / src_h
    with Image.open(trg_image) as img:
        trg_w, trg_h = img.size
        # trg_image, _ = pad_image(img)
        trg_image = img.resize((img_size, img_size)).convert('RGB')
        trg_x_scale, trg_y_scale = img_size / trg_w, img_size / trg_h
    
    src_points = [[int(np.round(x * src_x_scale)), int(np.round(y * src_y_scale))] for (x, y) in src_points]

    src_tensor = (PILToTensor()(src_image) / 255.0 - 0.5) * 2
    trg_tensor = (PILToTensor()(trg_image) / 255.0 - 0.5) * 2
    src_ft = dift.forward(src_tensor, prompt=src_prompt, ensemble_size=ensemble_size)
    trg_ft = dift.forward(trg_tensor, prompt=trg_prompt, ensemble_size=ensemble_size)
    num_channel = src_ft.size(1)
    cos = nn.CosineSimilarity(dim=1)
    
    
    src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)
    src_vectors = [src_ft[0, :, y, x].view(1, num_channel, 1, 1) for (x, y) in src_points]
    del src_ft
    gc.collect()
    torch.cuda.empty_cache()
    
    trg_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(trg_ft)
    cos_maps = [cos(src_vec, trg_ft).cpu().numpy() for src_vec in src_vectors]
    cos_values = [cos_map.max() for cos_map in cos_maps]
    # cos_maps = [crop_array(cos(src_vec, trg_ft).cpu().numpy(), trg_h * trg_x_scale, trg_w * trg_y_scale) for src_vec in src_vectors]

    del trg_ft
    gc.collect()
    torch.cuda.empty_cache()
    
    for cos_map in cos_maps:
        max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)[1:]
        trg_points.append([max_yx[1], max_yx[0]])
    
    trg_points = [[int(np.round(x / trg_x_scale)), int(np.round(y / trg_y_scale))] for (x, y) in trg_points]
    cos_maps = [nn.Upsample(size=(trg_h, trg_w), mode='bilinear')(torch.tensor(cos_map)[None, :, :, :]).numpy()[0] for cos_map in cos_maps] if return_cos_maps else None
    return trg_points, src_points, cos_maps, src_image, cos_values

