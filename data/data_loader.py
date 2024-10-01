import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.segmentation import slic
from skimage.measure import regionprops

def load_all_folds(base_path):
    folds = ["Fold 1", "Fold 2", "Fold 3"]
    all_images, all_masks, all_types = [], [], []
    
    for fold in folds:
        fold_path = os.path.join(base_path, fold)
        images = np.load(os.path.join(fold_path, "images", f"fold{fold[-1]}", "images.npy"))
        masks = np.load(os.path.join(fold_path, "masks", f"fold{fold[-1]}", "masks.npy"))
        types = np.load(os.path.join(fold_path, "images", f"fold{fold[-1]}", "types.npy"))
        
        all_images.append(images)
        all_masks.append(masks)
        all_types.append(types)
    
    return all_images, all_masks, all_types

def create_train_val_test_split(all_images, all_masks, all_types):
    splits = []
    
    for test_fold in range(3):
        train_val_folds = [i for i in range(3) if i != test_fold]
        
        train_val_images = np.concatenate([all_images[i] for i in train_val_folds])
        train_val_masks = np.concatenate([all_masks[i] for i in train_val_folds])
        train_val_types = np.concatenate([all_types[i] for i in train_val_folds])
        
        num_samples = len(train_val_images)
        num_val = num_samples // 10
        
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        split = {
            'train': {
                'images': train_val_images[train_indices],
                'masks': train_val_masks[train_indices],
                'types': train_val_types[train_indices]
            },
            'val': {
                'images': train_val_images[val_indices],
                'masks': train_val_masks[val_indices],
                'types': train_val_types[val_indices]
            },
            'test': {
                'images': all_images[test_fold],
                'masks': all_masks[test_fold],
                'types': all_types[test_fold]
            }
        }
        
        splits.append(split)
    
    return splits

class CellSegmentationDataset(Dataset):
    def __init__(self, images, masks, types, image_transform=None, mask_transform=None, augment=False):
        self.images = images
        self.masks = masks
        self.types = types
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augment = augment
        
        # Get unique tissue types and create a mapping
        self.unique_tissue_types = np.unique(self.types)
        self.tissue_type_to_idx = {t: i for i, t in enumerate(self.unique_tissue_types)}

        # Get unique cell types (excluding background)
        self.unique_cell_types = np.arange(self.masks[0].shape[-1] - 1)  # Exclude last channel (background)
        self.cell_type_to_idx = {t: i for i, t in enumerate(self.unique_cell_types)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        tissue_type = self.types[idx]

        # Normalize image to 0-1 range and convert to float32
        image = ((image - image.min()) / (image.max() - image.min())).astype(np.float32)
        
        # Create binary mask from the last channel (background) and convert to float32
        binary_mask = (mask[..., -1] == 0).astype(np.float32)  # Invert the background mask

        # Create multi-class mask for cell types (all channels except the last one)
        multi_class_mask = mask[..., :-1].astype(np.float32)
        multi_class_mask = np.divide(multi_class_mask, np.maximum(np.max(multi_class_mask), 1e-8))  # Normalize to 0-1 range

        # Apply binary mask to multi-class mask to exclude background
        multi_class_mask = multi_class_mask * binary_mask[..., np.newaxis]

        # Generate HV maps
        hv_map = self.generate_hv_map(binary_mask)

        # Create global cell labels
        global_cell_labels = np.zeros(len(self.unique_cell_types), dtype=np.float32)
        for i in range(multi_class_mask.shape[-1]):
            if np.any(multi_class_mask[..., i] > 0):
                global_cell_labels[i] = 1

        if self.augment:
            image, binary_mask, multi_class_mask, hv_map = self.apply_augmentation(image, binary_mask, multi_class_mask, hv_map)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            binary_mask = self.mask_transform(binary_mask)
            multi_class_mask = self.mask_transform(multi_class_mask)
            hv_map = self.mask_transform(hv_map)
        else:
            # If no mask_transform, convert to tensor manually
            binary_mask = torch.from_numpy(binary_mask).unsqueeze(0)
            multi_class_mask = torch.from_numpy(multi_class_mask).permute(2, 0, 1)
            hv_map = torch.from_numpy(hv_map).permute(2, 0, 1)

        # Create one-hot encoded tensor for tissue type
        tissue_type_idx = self.tissue_type_to_idx[tissue_type]
        tissue_type_onehot = torch.zeros(len(self.unique_tissue_types))
        tissue_type_onehot[tissue_type_idx] = 1

        # Convert global cell labels to tensor
        global_cell_labels = torch.from_numpy(global_cell_labels)

        return image, binary_mask, multi_class_mask, hv_map, tissue_type_onehot, global_cell_labels

    def generate_hv_map(self, binary_mask):
        label_img = (binary_mask * 255).astype(np.uint8)
        label_img = cv2.connectedComponents(label_img)[1]
        
        h_map = np.zeros_like(binary_mask, dtype=np.float32)
        v_map = np.zeros_like(binary_mask, dtype=np.float32)

        for region in regionprops(label_img):
            coords = region.coords
            center = region.centroid
            
            h_map[coords[:, 0], coords[:, 1]] = (coords[:, 1] - center[1]) / (region.bbox[3] - region.bbox[1] + 1e-5)
            v_map[coords[:, 0], coords[:, 1]] = (coords[:, 0] - center[0]) / (region.bbox[2] - region.bbox[0] + 1e-5)

        hv_map = np.stack([h_map, v_map], axis=-1)
        return hv_map

    def apply_augmentation(self, image, binary_mask, multi_class_mask, hv_map):
        # Convert numpy arrays to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)
        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0)
        multi_class_mask = torch.from_numpy(multi_class_mask).permute(2, 0, 1)
        hv_map = torch.from_numpy(hv_map).permute(2, 0, 1)

        original_size = image.shape[1:]

        # Random 90-degree rotation
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, [1, 2])
            binary_mask = torch.rot90(binary_mask, k, [1, 2])
            multi_class_mask = torch.rot90(multi_class_mask, k, [1, 2])
            hv_map = torch.rot90(hv_map, k, [1, 2])

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            binary_mask = TF.hflip(binary_mask)
            multi_class_mask = TF.hflip(multi_class_mask)
            hv_map = TF.hflip(hv_map)
            hv_map[0] = -hv_map[0]  # Flip horizontal map

        # Random vertical flip
        if torch.rand(1) < 0.5:
            image = TF.vflip(image)
            binary_mask = TF.vflip(binary_mask)
            multi_class_mask = TF.vflip(multi_class_mask)
            hv_map = TF.vflip(hv_map)
            hv_map[1] = -hv_map[1]  # Flip vertical map

        # Random scaling (downscaling)
        if torch.rand(1) < 0.5:
            scale_factor = torch.FloatTensor(1).uniform_(0.8, 1.0).item()
            new_size = [max(224, int(s * scale_factor)) for s in image.shape[1:]]
            image = TF.resize(image, new_size)
            binary_mask = TF.resize(binary_mask, new_size)
            multi_class_mask = TF.resize(multi_class_mask, new_size, interpolation=TF.InterpolationMode.NEAREST)

        # Elastic transformation
        if torch.rand(1) < 0.5:
            image = self.elastic_transform(image.permute(1, 2, 0).numpy())
            binary_mask = self.elastic_transform(binary_mask.squeeze().numpy(), is_mask=True)
            multi_class_mask = self.elastic_transform(multi_class_mask.permute(1, 2, 0).numpy(), is_mask=True)
            image = torch.from_numpy(image).permute(2, 0, 1)
            binary_mask = torch.from_numpy(binary_mask).unsqueeze(0)
            multi_class_mask = torch.from_numpy(multi_class_mask).permute(2, 0, 1)

        # Ensure image is large enough for subsequent operations
        if min(image.shape[1:]) < 224:
            scale_factor = 224 / min(image.shape[1:])
            new_size = [int(s * scale_factor) for s in image.shape[1:]]
            image = TF.resize(image, new_size)
            binary_mask = TF.resize(binary_mask, new_size)
            multi_class_mask = TF.resize(multi_class_mask, new_size, interpolation=TF.InterpolationMode.NEAREST)

        # Blurring
        if torch.rand(1) < 0.5:
            sigma = torch.FloatTensor(1).uniform_(0.1, 2.0).item()
            image = torch.from_numpy(gaussian_filter(image.numpy(), sigma=(0, sigma, sigma)))

        # Gaussian noise
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(image) * 0.1
            image = image + noise
            image = torch.clamp(image, 0, 1)

        # Color jittering
        if torch.rand(1) < 0.5:
            brightness_factor = torch.tensor(1.0).uniform_(0.8, 1.2).item()
            contrast_factor = torch.tensor(1.0).uniform_(0.8, 1.2).item()
            saturation_factor = torch.tensor(1.0).uniform_(0.8, 1.2).item()
            hue_factor = torch.tensor(1.0).uniform_(-0.1, 0.1).item()
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)
            image = TF.adjust_saturation(image, saturation_factor)
            image = TF.adjust_hue(image, hue_factor)

        # SLIC superpixels
        if torch.rand(1) < 0.5:
            image = self.apply_slic(image)

        # Zoom blur
        if torch.rand(1) < 0.5:
            image = self.zoom_blur(image)

        # Random cropping with resizing
        if torch.rand(1) < 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
            image = TF.crop(image, i, j, h, w)
            binary_mask = TF.crop(binary_mask, i, j, h, w)
            multi_class_mask = TF.crop(multi_class_mask, i, j, h, w)

        # Resize back to original size
        image = TF.resize(image, original_size)
        binary_mask = TF.resize(binary_mask, original_size)
        multi_class_mask = TF.resize(multi_class_mask, original_size, interpolation=TF.InterpolationMode.NEAREST)
        hv_map = TF.resize(hv_map, original_size, interpolation=TF.InterpolationMode.BILINEAR)

        return image, binary_mask, multi_class_mask, hv_map

# Usage example
def get_dataloaders(base_path, batch_size=16):
    all_images, all_masks, all_types = load_all_folds(base_path)
    data_splits = create_train_val_test_split(all_images, all_masks, all_types)

    image_transform = transforms.Compose([
        transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else torch.from_numpy(x).permute(2, 0, 1)),
        transforms.Lambda(lambda x: x.float())
    ])

    mask_transform = transforms.Compose([
        transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else torch.from_numpy(x).unsqueeze(0)),
        transforms.Lambda(lambda x: x.float())
    ])

    chosen_split = 2  # You can change this to use different splits

    train_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['train']['images'],
        data_splits[chosen_split]['train']['masks'],
        data_splits[chosen_split]['train']['types'],
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=True
    )

    val_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['val']['images'],
        data_splits[chosen_split]['val']['masks'],
        data_splits[chosen_split]['val']['types'],
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False
    )

    test_dataset = CellSegmentationDataset(
        data_splits[chosen_split]['test']['images'],
        data_splits[chosen_split]['test']['masks'],
        data_splits[chosen_split]['test']['types'],
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader