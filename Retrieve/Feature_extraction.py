import os
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "/home/ /.cache/huggingface/hub/model--openai--clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def get_labels(files):
    labels = []
    for file_path in files:
        directory = os.path.dirname(file_path)
        label = os.path.basename(directory)
        if label not in labels:
            labels.append(label)
    return labels

def list_files(dataset_path):
    images = []
    valid_images = [".jpg",".gif",".png",".jpeg"]
    for root, _, files in os.walk(dataset_path):
        for name in files:
            if os.path.splitext(name)[1].lower() in valid_images:
                images.append(os.path.join(root, name))
    return images

class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = list_files(self.img_dir)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = os.path.basename(os.path.dirname(img_path))  # get the parent directory name
        return image, img_path


dir_path = ""

# Iterate over each subdirectory in the main directory
for sub_dir in os.listdir(dir_path):
    full_sub_dir_path = os.path.join(dir_path, sub_dir) 
    if os.path.isdir(full_sub_dir_path):
        dataset = CustomImageDataset(full_sub_dir_path)
        train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True,)

        final_img_features = []
        final_img_filepaths = []
        for image_tensors, file_paths in tqdm(train_dataloader):
            try:
                image_features = model.get_image_features(image_tensors) #512
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.tolist()
                final_img_features.extend(image_features)
                final_img_filepaths.extend((list(file_paths)))
            except Exception as e:
                print("Exception occurred: ",e)
                break

        # Create a unique h5 filename for each sub-directory
        h5_filename = f"/home/ /shared/biggest/{sub_dir}_features.h5"
        with h5py.File(h5_filename, 'w') as h5f:
            h5f.create_dataset(f"{sub_dir}_features", data= np.array(final_img_features))
            # to save file names strings in byte format.
            h5f.create_dataset(f"{sub_dir}_filenames", data= np.array(final_img_filepaths, dtype=object))