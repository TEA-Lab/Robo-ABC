import gc
import torch
import numpy as np
from PIL import Image


def get_cor_pairs(extractor, src_image: str, trg_image: str, src_points: list, image_size: int = 512, layer: int = 9, 
                  facet: str = 'key', bin: bool = True, transpose_img_func=lambda x:x, transpose_pts_func = lambda x, y: (x, y),device='cuda:0'):

    # extracting descriptors for each image
    with Image.open(src_image) as img:
        src_image = transpose_img_func(img)
        src_image_width, src_image_height = img.size 
        src_image = src_image.resize((image_size, image_size)).convert('RGB')
        src_points = [transpose_pts_func(x, y) for x, y in src_points]
        src_x_scale, src_y_scale = image_size / src_image_width, image_size / src_image_height
        src_points = [[int(np.round(x * src_x_scale)), int(np.round(y * src_y_scale))] for (x, y) in src_points]

    with Image.open(trg_image) as img:
        trg_image_width, trg_image_height = img.size
        # trg_image, _ = pad_image(img)
        trg_image = img.resize((image_size, image_size)).convert('RGB')
        trg_x_scale, trg_y_scale = image_size / trg_image_width, image_size / trg_image_height
    
    src_image_batch, src_image_pil = extractor.preprocess(src_image, image_size)
    descriptors_src = extractor.extract_descriptors(src_image_batch.to(device), layer, facet, bin)

    num_patches_src, _ = extractor.num_patches, extractor.load_size

    indices_to_show = []
    for i in range(len(src_points)):
        transferred_x1 = (src_points[i][0] - extractor.stride[1] - extractor.p // 2)/extractor.stride[1] + 1
        transferred_y1 = (src_points[i][1] - extractor.stride[0] - extractor.p // 2)/extractor.stride[0] + 1
        indices_to_show.append(int(transferred_y1) * num_patches_src[1] + int(transferred_x1))
    
    descriptors_src_vec = descriptors_src[:, :, torch.Tensor(indices_to_show).to(torch.long)]

    del descriptors_src, src_image_batch
    gc.collect()
    torch.cuda.empty_cache()

    trg_image_batch, _ = extractor.preprocess(trg_image, image_size)
    descriptors_trg = extractor.extract_descriptors(trg_image_batch.to(device), layer, facet, bin)
    num_patches_trg, _ = extractor.num_patches, extractor.load_size

    # calculate similarity between src_image and trg_image descriptors
    similarities = chunk_cosine_sim(descriptors_src_vec, descriptors_trg)

    # calculate best buddies
    sim_src, nn_src = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_src, nn_src = sim_src[0, 0], nn_src[0, 0]

    del descriptors_trg, descriptors_src_vec, similarities, trg_image_batch
    gc.collect()
    torch.cuda.empty_cache()

    trg_img_indices_to_show = nn_src
    sim_values = sim_src.detach().cpu().numpy()
    # coordinates in descriptor map's dimensions
    trg_img_y_to_show = (trg_img_indices_to_show / num_patches_trg[1]).cpu().numpy()
    trg_img_x_to_show = (trg_img_indices_to_show % num_patches_trg[1]).cpu().numpy()
    trg_points = []
    for y, x in zip(trg_img_y_to_show, trg_img_x_to_show):
        x_trg_show = (int(x) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y_trg_show = (int(y) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        trg_points.append([y_trg_show, x_trg_show])
    trg_points = [[int(np.round(x / trg_x_scale)), int(np.round(y / trg_y_scale))] for (x, y) in trg_points]
    return trg_points, src_points, None, src_image, sim_values

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)
