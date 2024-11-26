# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import abc
from PIL import Image
from sc_models.ldm_sc.ptp_utils import diffusion_step, register_attention_control
import torch.nn.functional as F

import torch.nn as nn


def load_ldm(device, type="CompVis/stable-diffusion-v1-4"):

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    
    MY_TOKEN = ''
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    scheduler.set_timesteps(NUM_DDIM_STEPS)
    
    ldm = StableDiffusionPipeline.from_pretrained(type, use_auth_token=MY_TOKEN, scheduler=scheduler, local_files_only=True).to(device)

    for param in ldm.vae.parameters():
        param.requires_grad = False
    for param in ldm.text_encoder.parameters():
        param.requires_grad = False
    for param in ldm.unet.parameters():
        param.requires_grad = False
        
    return ldm
        

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):

        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return  0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):

        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):

        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):

        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def init_prompt(model, prompt: str):
    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=model.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(
        [prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    prompt = prompt
    
    return context, prompt

def init_random_noise(device, num_words = 77):
    return torch.randn(1, num_words, 768).to(device)

def image2latent(model, image, device):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            # print the max and min values of the image
            image = torch.from_numpy(image).float() * 2 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            latents = model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents


def reshape_attention(attention_map):
    """takes average over 0th dimension and reshapes into square image

    Args:
        attention_map (4, img_size, -1): _description_
    """
    attention_map = attention_map.mean(0)
    img_size = int(np.sqrt(attention_map.shape[0]))
    attention_map = attention_map.reshape(img_size, img_size, -1)
    return attention_map

def visualize_attention_map(attention_map, file_name):
    # save attention map
    attention_map = attention_map.unsqueeze(-1).repeat(1, 1, 3)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    attention_map = attention_map.detach().cpu().numpy()
    attention_map = (attention_map * 255).astype(np.uint8)
    img = Image.fromarray(attention_map)
    img.save(file_name)


@torch.no_grad()
def run_image_with_tokens_cropped(ldm, image, tokens, device='cuda', from_where = ["down_cross", "mid_cross", "up_cross"], index=0, upsample_res=512, noise_level=10, layers=[0, 1, 2, 3, 4, 5], num_iterations=20, crop_percent=100.0, image_mask = None):
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()
    
    num_samples = torch.zeros(len(layers), 4, 512, 512).to(device)
    sum_samples = torch.zeros(len(layers), 4, 512, 512).to(device)
    
    pixel_locs = torch.tensor([[0, 0], [0, 512], [512, 0], [512, 512]]).float().to(device)
    
    collected_attention_maps = []
    
    for i in range(num_iterations):
        
        if i < 4:
            pixel_loc = pixel_locs[i]
        else:
            
            _attention_maps = sum_samples/num_samples
            
            # remove all the nans
            _attention_maps[_attention_maps != _attention_maps] = 0
            
            _attention_maps = torch.mean(_attention_maps, dim=0)
            _attention_maps = torch.mean(_attention_maps, dim=0)
            
            max_val = find_max_pixel_value(_attention_maps, img_size = 512)+0.5
            
            pixel_loc = max_val.clone()
        
        cropped_image, cropped_pixel, y_start, height, x_start, width = crop_image(image, pixel_loc, crop_percent = crop_percent)
                
        latents = image2latent(ldm, cropped_image, device)
    
        controller = AttentionStore()
            
        register_attention_control(ldm, controller)
        
        latents = ldm.scheduler.add_noise(latents, torch.rand_like(latents), ldm.scheduler.timesteps[-3])
        
        latents = diffusion_step(ldm, controller, latents, tokens, ldm.scheduler.timesteps[-3], cfg=False)
        
        assert height == width
        
        _attention_maps = upscale_to_img_size(controller, from_where = from_where, upsample_res=height, layers=layers)
        
        num_samples[:, :, y_start:y_start+height, x_start:x_start+width] += 1
        sum_samples[:, :, y_start:y_start+height, x_start:x_start+width] += _attention_maps
        
        _attention_maps = sum_samples/num_samples
        
        if image_mask is not None:
            _attention_maps = _attention_maps * image_mask[None, None].to(device)
        
        collected_attention_maps.append(_attention_maps.clone())
        
    # visualize sum_samples/num_samples
    attention_maps = sum_samples/num_samples
    
    if image_mask is not None:
        attention_maps = attention_maps * image_mask[None, None].to(device)
    
    return attention_maps, collected_attention_maps    


def upscale_to_img_size(controller, from_where = ["down_cross", "mid_cross", "up_cross"], upsample_res=512, layers=[0, 1, 2, 3, 4, 5]):
    """
    returns the bilinearly upsampled attention map of size upsample_res x upsample_res for the first word in the prompt
    """
    
    attention_maps = controller.get_average_attention()
    
    imgs = []
    
    layer_overall = -1
    
    for key in from_where:
        for layer in range(len(attention_maps[key])):
            
            layer_overall += 1
            
            
            if layer_overall not in layers:
                continue
                
            img = attention_maps[key][layer]
            
            img = img.reshape(4, int(img.shape[1]**0.5), int(img.shape[1]**0.5), img.shape[2])[None, :, :, :, 1]
            
            if upsample_res != -1:
                # bilinearly upsample the image to img_sizeximg_size
                img = F.interpolate(img, size=(upsample_res, upsample_res), mode='bilinear', align_corners=False)

            imgs.append(img)

    imgs = torch.cat(imgs, dim=0)
    
    return imgs


def softargmax2d(input, beta = 1000):
    *_, h, w = input.shape
    
    assert h == w, "only square images are supported"

    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(input*beta, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).to(input.device).float()
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).to(input.device).float()

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_c, result_r], dim=-1)

    return result/h


def find_context(image, ldm, pixel_loc, context_estimator, device='cuda'):
    
    with torch.no_grad():
        latent = image2latent(ldm, image.numpy().transpose(1, 2, 0), device)
        
    context = context_estimator(latent, pixel_loc)
    
    return context
    

def find_max_pixel_value(tens, img_size=512, ignore_border = True):
    """finds the 2d pixel location that is the max value in the tensor

    Args:
        tens (tensor): shape (height, width)
    """
    
    assert len(tens.shape) == 2, "tens must be 2d"
    
    _tens = tens.clone()
    height = _tens.shape[0]
    
    _tens = _tens.reshape(-1)
    max_loc = torch.argmax(_tens)
    max_pixel = torch.stack([max_loc % height, torch.div(max_loc, height, rounding_mode='floor')])
    
    max_pixel = max_pixel/height*img_size
    
    return max_pixel

def visualize_image_with_points(image, point, name, save_folder = "outputs"):
    
    """The point is in pixel numbers
    """
    
    import matplotlib.pyplot as plt
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        try:
            image = image.permute(1, 2, 0).detach().cpu().numpy()
        except:
            import ipdb; ipdb.set_trace()   
    
    
    # make the figure without a border
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10, 10)
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    plt.imshow(image, aspect='auto')
    
    if point is not None:
        # plot point on image
        plt.scatter(point[0].cpu(), point[1].cpu(), s=20, marker='o', c='r')
    
    
    plt.savefig(f'{save_folder}/{name}.png', dpi=200)
    plt.close()


def gaussian_circle(pos, size=64, sigma=16, device = "cuda"):
    """Create a 2D Gaussian circle with a given size, standard deviation, and center coordinates.
    
    pos is in between 0 and 1
    
    """
    _pos = pos*size
    grid = torch.meshgrid(torch.arange(size).to(device), torch.arange(size).to(device), indexing='ij')
    grid = torch.stack(grid, dim=-1)
    dist_sq = (grid[..., 1] - _pos[0])**2 + (grid[..., 0] - _pos[1])**2
    dist_sq = -1*dist_sq / (2. * sigma**2.)
    gaussian = torch.exp(dist_sq)
    return gaussian


def crop_image(image, pixel, crop_percent=80, margin=0.15):
    
    """pixel is an integer between 0 and image.shape[1] or image.shape[2]
    """
    
    assert 0 < crop_percent <= 100, "crop_percent should be between 0 and 100"

    height, width, channels = image.shape
    crop_height = int(height * crop_percent / 100)
    crop_width = int(width * crop_percent / 100)

    # Calculate the crop region's top-left corner
    x, y = pixel
    
    # Calculate safe margin
    safe_margin_x = int(crop_width * margin)
    safe_margin_y = int(crop_height * margin)
    
    x_start_min = max(0, x - crop_width + safe_margin_x)
    x_start_min = min(x_start_min, width - crop_width)
    x_start_max = max(0, x - safe_margin_x)
    x_start_max = min(x_start_max, width - crop_width)
    
    y_start_min = max(0, y - crop_height + safe_margin_y)
    y_start_min = min(y_start_min, height - crop_height)
    y_start_max = max(0, y - safe_margin_y)
    y_start_max = min(y_start_max, height - crop_height)

    # Choose a random top-left corner within the allowed bounds
    x_start = torch.randint(int(x_start_min), int(x_start_max) + 1, (1,)).item()
    y_start = torch.randint(int(y_start_min), int(y_start_max) + 1, (1,)).item()

    # Crop the image
    cropped_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]
    
    # bilinearly upsample to 512x512
    cropped_image = torch.nn.functional.interpolate(torch.tensor(cropped_image[None]).permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=False)[0]
    
    # calculate new pixel location
    new_pixel = torch.stack([x-x_start, y-y_start])
    new_pixel = new_pixel/crop_width

    return cropped_image.permute(1, 2, 0).numpy(), new_pixel, y_start, crop_height, x_start, crop_width


def optimize_prompt(ldm, image, pixel_loc, context=None, device="cuda", num_steps=100, from_where = ["down_cross", "mid_cross", "up_cross"], upsample_res = 32, layers = [0, 1, 2, 3, 4, 5], lr=1e-3, noise_level = -1, sigma = 32, flip_prob = 0.5, crop_percent=80):
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()
        
    if context is None:
        context = init_random_noise(device)
        
    context.requires_grad = True
    
    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=lr)
    
    # time the optimization
    import time
    start = time.time()
    
    for iteration in range(num_steps):
        
        with torch.no_grad():
        
            if np.random.rand() > flip_prob:
                
                cropped_image, cropped_pixel, _, _, _, _ = crop_image(image, pixel_loc*512, crop_percent = crop_percent)
                
                latent = image2latent(ldm, cropped_image, device)
                
                _pixel_loc = cropped_pixel.clone()
            else:
                
                image_flipped = np.flip(image, axis=1).copy()
                
                pixel_loc_flipped = pixel_loc.clone()
                # flip pixel loc
                pixel_loc_flipped[0] = 1 - pixel_loc_flipped[0]
                
                cropped_image, cropped_pixel, _, _, _, _ = crop_image(image_flipped, pixel_loc_flipped*512, crop_percent = crop_percent)
                
                _pixel_loc = cropped_pixel.clone()
                
                latent = image2latent(ldm, cropped_image, device)
            
        noisy_image = ldm.scheduler.add_noise(latent, torch.rand_like(latent), ldm.scheduler.timesteps[noise_level])
        
        controller = AttentionStore()
        
        register_attention_control(ldm, controller)
        
        _ = diffusion_step(ldm, controller, noisy_image, context, ldm.scheduler.timesteps[noise_level], cfg = False)
        
        attention_maps = upscale_to_img_size(controller, from_where = from_where, upsample_res=upsample_res, layers = layers)
        num_maps = attention_maps.shape[0]
        
        # divide by the mean along the dim=1
        attention_maps = torch.mean(attention_maps, dim=1)

        gt_maps = gaussian_circle(_pixel_loc, size=upsample_res, sigma=sigma, device = device)
        
        gt_maps = gt_maps.reshape(1, -1).repeat(num_maps, 1)
        attention_maps = attention_maps.reshape(num_maps, -1)
        
        loss = torch.nn.MSELoss()(attention_maps, gt_maps)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return context

