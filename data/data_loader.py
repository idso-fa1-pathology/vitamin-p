import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import cv2
from skimage.segmentation import slic
from scipy.ndimage import map_coordinates
from torchvision.transforms import functional as TF

def load_tissuenet_data(base_path):
    datasets = ['train', 'val', 'test']
    data = {}
    
    for dataset in datasets:
        file_path = os.path.join(base_path, f'{dataset}.npz')
        loaded_data = np.load(file_path)
        data[dataset] = {
            'images': loaded_data['X'],
            'masks': loaded_data['y']
        }
    
    return data

def create_256_crops(images, masks, crop_size=256, samples_per_image=4):
    crops_images, crops_masks = [], []
    
    for img, mask in zip(images, masks):
        h, w = img.shape[:2]
        if h == w == crop_size:
            crops_images.append(img)
            crops_masks.append(mask)
        else:
            for _ in range(samples_per_image):
                top = np.random.randint(0, h - crop_size + 1)
                left = np.random.randint(0, w - crop_size + 1)
                
                img_crop = img[top:top+crop_size, left:left+crop_size]
                mask_crop = mask[top:top+crop_size, left:left+crop_size]
                
                crops_images.append(img_crop)
                crops_masks.append(mask_crop)
    
    return np.array(crops_images), np.array(crops_masks)

def create_test_crops(images, masks, crop_size=256):
    crops_images, crops_masks = [], []
    
    for img, mask in zip(images, masks):
        h, w = img.shape[:2]
        if h == w == crop_size:
            crops_images.append(img)
            crops_masks.append(mask)
        else:
            for top in range(0, h - crop_size + 1, crop_size):
                for left in range(0, w - crop_size + 1, crop_size):
                    img_crop = img[top:top+crop_size, left:left+crop_size]
                    mask_crop = mask[top:top+crop_size, left:left+crop_size]
                    
                    crops_images.append(img_crop)
                    crops_masks.append(mask_crop)
    
    return np.array(crops_images), np.array(crops_masks)

class TissueNetDataset(Dataset):
    def __init__(self, images, masks, image_transform=None, mask_transform=None, hv_transform=None, augment=False):
        self.images = images
        self.masks = masks
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.hv_transform = hv_transform
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image_min, image_max = image.min(), image.max()
        image = ((image - image_min) / (image_max - image_min + 1e-8)).astype(np.float32)

        cell_mask = mask[..., 0].astype(np.int32)
        nuclei_mask = mask[..., 1].astype(np.int32)

        cell_hv_map = self.generate_hv_map(cell_mask)
        nuclei_hv_map = self.generate_hv_map(nuclei_mask)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            cell_mask = self.mask_transform(cell_mask)
            nuclei_mask = self.mask_transform(nuclei_mask)

        if self.hv_transform:
            cell_hv_map = self.hv_transform(cell_hv_map)
            nuclei_hv_map = self.hv_transform(nuclei_hv_map)

        return image, cell_mask, nuclei_mask, cell_hv_map, nuclei_hv_map

    def generate_hv_map(self, instance_mask):
        h_map = np.zeros_like(instance_mask, dtype=np.float32)
        v_map = np.zeros_like(instance_mask, dtype=np.float32)

        for instance_id in np.unique(instance_mask):
            if instance_id == 0:
                continue

            instance_pixels = np.where(instance_mask == instance_id)
            y_coords, x_coords = instance_pixels
            center_y, center_x = np.mean(y_coords), np.mean(x_coords)

            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            height = max_y - min_y + 1
            width = max_x - min_x + 1

            norm_factor = max(height, width)

            h_map[instance_pixels] = (x_coords - center_x) / norm_factor
            v_map[instance_pixels] = (y_coords - center_y) / norm_factor

        hv_map = np.stack([h_map, v_map], axis=-1)
        return hv_map

    # ... (other methods like apply_augmentation, elastic_transform, apply_slic, zoom_blur)

# Define transforms
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256)
])

mask_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x).unsqueeze(0).long()),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)
])

hv_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256)
])

def get_data_loaders(base_path, batch_size=16):
    tissuenet_data = load_tissuenet_data(base_path)
    
    train_crops_images, train_crops_masks = create_256_crops(tissuenet_data['train']['images'], tissuenet_data['train']['masks'])
    val_crops_images, val_crops_masks = create_256_crops(tissuenet_data['val']['images'], tissuenet_data['val']['masks'])
    test_crops_images, test_crops_masks = create_test_crops(tissuenet_data['test']['images'], tissuenet_data['test']['masks'])

    train_dataset = TissueNetDataset(
        train_crops_images,
        train_crops_masks,
        image_transform=image_transform,
        mask_transform=mask_transform,
        hv_transform=hv_transform,
        augment=True
    )

    val_dataset = TissueNetDataset(
        val_crops_images,
        val_crops_masks,
        image_transform=image_transform,
        mask_transform=mask_transform,
        hv_transform=hv_transform,
        augment=False
    )

    test_dataset = TissueNetDataset(
        test_crops_images,
        test_crops_masks,
        image_transform=image_transform,
        mask_transform=mask_transform,
        hv_transform=hv_transform,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader