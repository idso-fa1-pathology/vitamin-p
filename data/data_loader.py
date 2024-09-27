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
    def apply_augmentation(self, image, cell_mask, nuclei_mask, cell_hv_map, nuclei_hv_map):
        # Convert numpy arrays to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)
        cell_mask = torch.from_numpy(cell_mask).unsqueeze(0)
        nuclei_mask = torch.from_numpy(nuclei_mask).unsqueeze(0)
        cell_hv_map = torch.from_numpy(cell_hv_map).permute(2, 0, 1)
        nuclei_hv_map = torch.from_numpy(nuclei_hv_map).permute(2, 0, 1)

        original_size = image.shape[1:]

        # Random 90-degree rotation
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, [1, 2])
            cell_mask = torch.rot90(cell_mask, k, [1, 2])
            nuclei_mask = torch.rot90(nuclei_mask, k, [1, 2])
            cell_hv_map = torch.rot90(cell_hv_map, k, [1, 2])
            nuclei_hv_map = torch.rot90(nuclei_hv_map, k, [1, 2])

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            cell_mask = TF.hflip(cell_mask)
            nuclei_mask = TF.hflip(nuclei_mask)
            cell_hv_map = TF.hflip(cell_hv_map)
            nuclei_hv_map = TF.hflip(nuclei_hv_map)
            cell_hv_map[0] = -cell_hv_map[0]  # Flip horizontal map
            nuclei_hv_map[0] = -nuclei_hv_map[0]  # Flip horizontal map

        # Random vertical flip
        if torch.rand(1) < 0.5:
            image = TF.vflip(image)
            cell_mask = TF.vflip(cell_mask)
            nuclei_mask = TF.vflip(nuclei_mask)
            cell_hv_map = TF.vflip(cell_hv_map)
            nuclei_hv_map = TF.vflip(nuclei_hv_map)
            cell_hv_map[1] = -cell_hv_map[1]  # Flip vertical map
            nuclei_hv_map[1] = -nuclei_hv_map[1]  # Flip vertical map

        # Random scaling (downscaling)
        if torch.rand(1) < 0.5:
            scale_factor = torch.FloatTensor(1).uniform_(0.8, 1.0).item()
            new_size = [max(224, int(s * scale_factor)) for s in image.shape[1:]]
            image = TF.resize(image, new_size)
            cell_mask = TF.resize(cell_mask, new_size, interpolation=TF.InterpolationMode.NEAREST)
            nuclei_mask = TF.resize(nuclei_mask, new_size, interpolation=TF.InterpolationMode.NEAREST)
            cell_hv_map = TF.resize(cell_hv_map, new_size)
            nuclei_hv_map = TF.resize(nuclei_hv_map, new_size)

        # Ensure image is large enough for subsequent operations
        if min(image.shape[1:]) < 224:
            scale_factor = 224 / min(image.shape[1:])
            new_size = [int(s * scale_factor) for s in image.shape[1:]]
            image = TF.resize(image, new_size)
            cell_mask = TF.resize(cell_mask, new_size, interpolation=TF.InterpolationMode.NEAREST)
            nuclei_mask = TF.resize(nuclei_mask, new_size, interpolation=TF.InterpolationMode.NEAREST)
            cell_hv_map = TF.resize(cell_hv_map, new_size)
            nuclei_hv_map = TF.resize(nuclei_hv_map, new_size)

        # Resize back to original size
        image = TF.resize(image, original_size)
        cell_mask = TF.resize(cell_mask, original_size, interpolation=TF.InterpolationMode.NEAREST)
        nuclei_mask = TF.resize(nuclei_mask, original_size, interpolation=TF.InterpolationMode.NEAREST)
        cell_hv_map = TF.resize(cell_hv_map, original_size)
        nuclei_hv_map = TF.resize(nuclei_hv_map, original_size)

        return image, cell_mask, nuclei_mask, cell_hv_map, nuclei_hv_map

    def elastic_transform(self, image, alpha=1, sigma=0.1, alpha_affine=0.1, is_mask=False):
        """Elastic deformation of images as described in [Simard2003]_."""
        random_state = np.random.RandomState(None)

        if image.ndim == 2:
            shape = image.shape
        elif image.ndim == 3:
            shape = image.shape[:2]
        else:
            raise ValueError("Image must be 2D or 3D")

        # Random affine
        center_square = np.float32(shape) // 2
        square_size = min(shape) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)

        if is_mask:
            if image.ndim == 2:
                image = cv2.warpAffine(image, M, shape[::-1], borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_NEAREST)
            else:
                image = np.stack([cv2.warpAffine(image[:,:,i], M, shape[::-1], borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_NEAREST) for i in range(image.shape[2])], axis=2)
        else:
            if image.ndim == 2:
                image = cv2.warpAffine(image, M, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            else:
                image = np.stack([cv2.warpAffine(image[:,:,i], M, shape[::-1], borderMode=cv2.BORDER_REFLECT_101) for i in range(image.shape[2])], axis=2)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        if is_mask:
            if image.ndim == 2:
                return map_coordinates(image, indices, order=0, mode='constant').reshape(shape)
            else:
                return np.stack([map_coordinates(image[:,:,i], indices, order=0, mode='constant').reshape(shape) for i in range(image.shape[2])], axis=2)
        else:
            if image.ndim == 2:
                return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
            else:
                return np.stack([map_coordinates(image[:,:,i], indices, order=1, mode='reflect').reshape(shape) for i in range(image.shape[2])], axis=2)

    def apply_slic(self, image):
        image_np = image.numpy().transpose(1, 2, 0)
        segments = slic(image_np, n_segments=100, compactness=10, sigma=1)
        out = np.zeros_like(image_np)
        for i in np.unique(segments):
            mask = segments == i
            out[mask] = np.mean(image_np[mask], axis=0)
        return torch.from_numpy(out.transpose(2, 0, 1))

    def zoom_blur(self, image, max_factor=1.2):
        c, h, w = image.shape
        zoom_factor = torch.FloatTensor(1).uniform_(1, max_factor).item()
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        zoom_image = TF.resize(image, (zh, zw))
        zoom_image = TF.center_crop(zoom_image, (h, w))
        return (image + zoom_image) / 2

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