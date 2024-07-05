import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from concurrent.futures import ThreadPoolExecutor


class MixtureDataset(Dataset):
    def __init__(self, folders, num_dataset, ratios, transform=None, resize_dim=(256, 256), pre_load=False, load_to_ram=True):
        assert len(folders) == len(ratios), "Each folder must have a corresponding ratio."
        assert sum(ratios) == 1, "Ratios must sum to 1."

        self.folders = folders
        self.num_dataset = num_dataset
        self.ratios = ratios
        self.resize_dim = resize_dim
        self.transform = transform
        self.pre_load =pre_load
        self.load_to_ram = load_to_ram
        self.image_cache = {}  # Initialize an empty cache

        # Calculate the total ratio to determine proportions
        images_per_folder = [int(ratio * num_dataset) for ratio in ratios]
        if sum(images_per_folder) < num_dataset:
            images_per_folder[-1] += num_dataset - sum(images_per_folder)

        self.images = []
        self.preloaded_images = []

        if pre_load:
            self._preload_images_multithreaded(folders, images_per_folder)
        else:
            self._prepare_image_paths(folders, images_per_folder)

    def _prepare_image_paths(self, folders, images_per_folder):
        for folder_idx, (folder_path, num_images) in enumerate(zip(folders, images_per_folder)):
            selected_images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)[:num_images]]
            self.images.extend([(img_path, folder_idx) for img_path in selected_images])

    def _load_and_process_image(self, img_path_folder_idx):
        img_path, folder_idx = img_path_folder_idx
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        if image.shape[:2] != self.resize_dim:
            image = cv2.resize(image, self.resize_dim, interpolation=cv2.INTER_AREA)  # Resize image
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([folder_idx], dtype=torch.float)  # Shape [1]
        return (image, label)

    def _preload_images_multithreaded(self, folders, images_per_folder):
        image_paths_labels = []
        for folder_idx, (folder_path, num_images) in enumerate(zip(folders, images_per_folder)):
            selected_images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)[:num_images]]
            image_paths_labels.extend([(img_path, folder_idx) for img_path in selected_images])

        with ThreadPoolExecutor() as executor:
            self.preloaded_images = list(executor.map(self._load_and_process_image, image_paths_labels))

    def __len__(self):
        return len(self.preloaded_images) if self.pre_load else len(self.images)

    def __getitem__(self, idx):
        if self.pre_load:
            # Return preloaded image and label
            return self.preloaded_images[idx]
        else:
            # Check cache first
            if idx in self.image_cache and self.load_to_ram:
                return self.image_cache[idx]
            else:
                # Load and transform the image on-the-fly
                img_path, folder_label = self.images[idx]
                image_label_pair = self._load_and_process_image((img_path, folder_label))
                # Store in cache
                self.image_cache[idx] = image_label_pair
                return image_label_pair



class ImagesFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None, resize_dim=(256, 256), num_images=None, pre_load=False, load_to_ram=True):
        self.folder_path = folder_path
        self.transform = transform
        self.pre_load = pre_load
        self.load_to_ram = load_to_ram
        self.resize_dim = resize_dim
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Initialize an empty cache
        self.image_cache = {}

        if num_images is not None:
            self.image_paths = self.image_paths[num_images[0]:num_images[1]]

        if self.pre_load:
            self.images = self.load_images_multithreaded()
        else:
            self.images = None

    def load_and_process_image(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        if image.shape[:2] != self.resize_dim:
            image = cv2.resize(image, self.resize_dim, interpolation=cv2.INTER_AREA)  # Resize image
        if self.transform:
            image = self.transform(image)
        return image

    def load_images_multithreaded(self):
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(self.load_and_process_image, self.image_paths))
        return images

    def __len__(self):
        return len(self.images) if self.pre_load else len(self.image_paths)

    def __getitem__(self, idx):
        pseudo_label = torch.tensor([0], dtype=torch.float)  # Assuming a dummy label for demonstration
        if self.pre_load:
            # Return preloaded and transformed image
            return self.images[idx], pseudo_label
        else:
            # Check if the image is in the cache
            if idx in self.image_cache and self.load_to_ram:
                # Return cached image
                image = self.image_cache[idx]
            else:
                # Load and transform image on-the-fly
                img_path = self.image_paths[idx]
                image = self.load_and_process_image(img_path)
                # Cache the image for future use
                self.image_cache[idx] = image
            return image, pseudo_label
    
class OffsetDataset(Dataset):
    def __init__(self, mix_dataset, offset_dataset):
        self.mix_dataset = mix_dataset
        self.offset_dataset = offset_dataset

    def __len__(self):
        return len(self.mix_dataset)

    def __getitem__(self, idx):
        mix_image, _ = self.mix_dataset[idx]
        offset_image, _ = self.offset_dataset[np.random.randint(len(self.offset_dataset))]
        return mix_image, torch.tensor([1], dtype=torch.float), offset_image, torch.tensor([0], dtype=torch.float)

class MixtureDataset_label(Dataset):
    def __init__(self, mix_dataset):
        self.mix_dataset = mix_dataset

    def __len__(self):
        return len(self.mix_dataset)

    def __getitem__(self, idx):
        mix_image, _ = self.mix_dataset[idx]
        return mix_image, idx

class OffsetDataset_label(Dataset):
    def __init__(self, mix_dataset, clean_dataset):
        self.mix_dataset = mix_dataset
        self.clean_dataset = clean_dataset

    def __len__(self):
        return len(self.mix_dataset)

    def __getitem__(self, idx):
        mix_image, _ = self.mix_dataset[idx]
        clean_image, _ = self.clean_dataset[np.random.randint(len(self.clean_dataset))]
        return mix_image, idx, clean_image, torch.tensor([0], dtype=torch.float)