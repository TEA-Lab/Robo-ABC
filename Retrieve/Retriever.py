import os
import shutil
import h5py
import faiss
import numpy as np
from PIL import Image

from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = ""
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with h5py.File('.h5', 'r') as h5f:
    all_features = np.array(h5f['all_features'])
    all_filenames = np.array(h5f['all_filenames'])

faiss_index = faiss.IndexFlatIP(all_features.shape[1])
faiss_index.add(all_features)

folder_path = ""
txt_folder_path = ""

for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(subdir, file)
            image = Image.open(img_path)
            t_image = transform(image)
            inputs = processor(images=t_image, return_tensors="pt")  

            query_features = model(**inputs).pooler_output 
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_features = query_features.detach().numpy()

            K_neighbours = 5
            distances, indices = faiss_index.search(query_features, K_neighbours)

            for index in range(K_neighbours):
                similar_image_filename = all_filenames[indices[0][index]]
                similar_image_filename_str = similar_image_filename.decode('utf-8')

                # Add suffix to filename
                filename, ext = os.path.splitext(similar_image_filename_str)
                new_filename = f'{filename}_top{index+1}{ext}'
                new_path = os.path.join(subdir, os.path.basename(new_filename))
                shutil.copy(similar_image_filename_str, new_path)

                txt_filename_without_path = os.path.splitext(os.path.basename(similar_image_filename_str))[0] + '.txt'
                for txt_subdir, txt_dirs, txt_files in os.walk(txt_folder_path):
                    if txt_filename_without_path in txt_files:
                        txt_filename = os.path.join(txt_subdir, txt_filename_without_path)

                        # Add suffix to txt file
                        txt_filename_without_ext, txt_ext = os.path.splitext(txt_filename)
                        new_txt_filename = f'{txt_filename_without_ext}_top{index+1}{txt_ext}'
                        new_txt_path = os.path.join(subdir, os.path.basename(new_txt_filename))
                        shutil.copy(txt_filename, new_txt_path)