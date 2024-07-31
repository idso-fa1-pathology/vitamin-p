import tensorflow as tf
import numpy as np
import os

def load_pannuke_fold(fold_path):
    fold_number = os.path.basename(fold_path).split()[-1]  # Extract the fold number
    images = np.load(os.path.join(fold_path, 'images', f'fold{fold_number}', 'images.npy'))
    masks = np.load(os.path.join(fold_path, 'masks', f'fold{fold_number}', 'masks.npy'))
    types = np.load(os.path.join(fold_path, 'images', f'fold{fold_number}', 'types.npy'))
    
    print(f"Raw images shape: {images.shape}")
    print(f"Raw masks shape: {masks.shape}")
    print(f"Raw types shape: {types.shape}")
    
    # Ensure images are float32 and in range [0, 1]
    images = images.astype(np.float32) / 255.0
    
    print(f"Processed images shape: {images.shape}")
    print(f"Processed masks shape: {masks.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Masks dtype: {masks.dtype}")
    print(f"Images min and max: {np.min(images):.4f}, {np.max(images):.4f}")
    print(f"Masks min and max: {np.min(masks):.4f}, {np.max(masks):.4f}")
    print(f"Percentage of non-zero mask pixels: {(masks > 0).mean() * 100:.2f}%")
    
    return images, masks, types

def create_hv_maps(masks):
    h, w = masks.shape[1:3]
    y_coords, x_coords = np.ogrid[:h, :w]
    hv_maps = np.zeros((masks.shape[0], h, w, 2), dtype=np.float32)
    
    for i in range(masks.shape[0]):
        for c in range(masks.shape[-1]):
            mask = masks[i, ..., c]
            if mask.sum() > 0:
                center_y, center_x = np.mean(np.where(mask), axis=1)
                hv_maps[i, ..., 0] += (x_coords - center_x) * mask
                hv_maps[i, ..., 1] += (y_coords - center_y) * mask
    
    # Normalize HV maps to [-1, 1]
    hv_maps /= np.maximum(h, w)
    return hv_maps

def load_fold(fold_path):
    fold_number = os.path.basename(fold_path).split()[-1]
    images = np.load(os.path.join(fold_path, 'images', f'fold{fold_number}', 'images.npy'))
    masks = np.load(os.path.join(fold_path, 'masks', f'fold{fold_number}', 'masks.npy'))
    types = np.load(os.path.join(fold_path, 'images', f'fold{fold_number}', 'types.npy'))
    
    # Convert images to float32 and normalize to [0, 1]
    images = images.astype(np.float32) / 255.0
    
    # Create binary masks for NP branch
    binary_masks = (masks.sum(axis=-1) > 0).astype(np.float32)
    
    # Create horizontal and vertical distance maps for HV branch
    hv_maps = create_hv_maps(masks)
    
    # Convert string labels to integer indices
    unique_types = np.unique(types)
    type_to_index = {t: i for i, t in enumerate(unique_types)}
    types_indices = np.array([type_to_index[t] for t in types])
    
    # Convert types to one-hot encoded format
    num_classes = len(unique_types)
    types_one_hot = tf.keras.utils.to_categorical(types_indices, num_classes=num_classes)
    
    return images, binary_masks, hv_maps, masks, types_one_hot, unique_types

def preprocess_pannuke_data(data_dir, fold, batch_size):
    all_images, all_binary_masks, all_hv_maps, all_masks, all_types, unique_types = [], [], [], [], [], None
    for fold_name in ['Fold 1', 'Fold 2', 'Fold 3']:
        fold_path = os.path.join(data_dir, fold_name)
        images, binary_masks, hv_maps, masks, types, fold_unique_types = load_fold(fold_path)
        all_images.append(images)
        all_binary_masks.append(binary_masks)
        all_hv_maps.append(hv_maps)
        all_masks.append(masks)
        all_types.append(types)
        if unique_types is None:
            unique_types = fold_unique_types

    all_images = np.concatenate(all_images)
    all_binary_masks = np.concatenate(all_binary_masks)
    all_hv_maps = np.concatenate(all_hv_maps)
    all_masks = np.concatenate(all_masks)
    all_types = np.concatenate(all_types)

    print(f"All images shape: {all_images.shape}")
    print(f"All binary masks shape: {all_binary_masks.shape}")
    print(f"All HV maps shape: {all_hv_maps.shape}")
    print(f"All masks shape: {all_masks.shape}")
    print(f"All types shape: {all_types.shape}")

    # Split data
    total_samples = len(all_images)
    if fold == 1:
        train_end = int(0.7 * total_samples)
        val_end = int(0.85 * total_samples)
        train = (all_images[:train_end], all_binary_masks[:train_end], all_hv_maps[:train_end], all_masks[:train_end], all_types[:train_end])
        val = (all_images[train_end:val_end], all_binary_masks[train_end:val_end], all_hv_maps[train_end:val_end], all_masks[train_end:val_end], all_types[train_end:val_end])
        test = (all_images[val_end:], all_binary_masks[val_end:], all_hv_maps[val_end:], all_masks[val_end:], all_types[val_end:])
    elif fold == 2:
        train_start = int(0.15 * total_samples)
        train_end = int(0.85 * total_samples)
        train = (all_images[train_start:train_end], all_binary_masks[train_start:train_end], all_hv_maps[train_start:train_end], all_masks[train_start:train_end], all_types[train_start:train_end])
        val = (all_images[:train_start], all_binary_masks[:train_start], all_hv_maps[:train_start], all_masks[:train_start], all_types[:train_start])
        test = (all_images[train_end:], all_binary_masks[train_end:], all_hv_maps[train_end:], all_masks[train_end:], all_types[train_end:])
    elif fold == 3:
        train_start = int(0.3 * total_samples)
        val_start = int(0.85 * total_samples)
        train = (all_images[train_start:], all_binary_masks[train_start:], all_hv_maps[train_start:], all_masks[train_start:], all_types[train_start:])
        val = (all_images[val_start:], all_binary_masks[val_start:], all_hv_maps[val_start:], all_masks[val_start:], all_types[val_start:])
        test = (all_images[:train_start], all_binary_masks[:train_start], all_hv_maps[:train_start], all_masks[:train_start], all_types[:train_start])
    else:
        raise ValueError("Invalid fold number. Choose 1, 2, or 3.")

    # Create TensorFlow datasets
    def create_dataset(images, binary_masks, hv_maps, masks, types):
        return tf.data.Dataset.from_tensor_slices((
            images,
            {
                'np_branch': binary_masks[..., np.newaxis],  # Add channel dimension
                'hv_branch': hv_maps,
                'nt_branch': masks,
                'tc_branch': types
            }
        )).shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_dataset = create_dataset(*train)
    val_dataset = create_dataset(*val)
    test_dataset = create_dataset(*test)

    # Print shapes for debugging
    for images, labels in train_dataset.take(1):
        print("Sample data:")
        print(f"Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"NP branch shape: {labels['np_branch'].shape}, dtype: {labels['np_branch'].dtype}")
        print(f"HV branch shape: {labels['hv_branch'].shape}, dtype: {labels['hv_branch'].dtype}")
        print(f"NT branch shape: {labels['nt_branch'].shape}, dtype: {labels['nt_branch'].dtype}")
        print(f"TC branch shape: {labels['tc_branch'].shape}, dtype: {labels['tc_branch'].dtype}")
        print(f"NP branch min/max: {tf.reduce_min(labels['np_branch'])}/{tf.reduce_max(labels['np_branch'])}")
        print(f"HV branch min/max: {tf.reduce_min(labels['hv_branch'])}/{tf.reduce_max(labels['hv_branch'])}")
        print(f"NT branch min/max: {tf.reduce_min(labels['nt_branch'])}/{tf.reduce_max(labels['nt_branch'])}")
        print(f"TC branch min/max: {tf.reduce_min(labels['tc_branch'])}/{tf.reduce_max(labels['tc_branch'])}")

    return train_dataset, val_dataset, test_dataset, unique_types

print("Updated preprocess_pannuke_data function")

def load_and_preprocess_pannuke(data_dir):
    folds = ['Fold 1', 'Fold 2', 'Fold 3']
    all_images, all_masks, all_types = [], [], []

    for fold in folds:
        fold_path = os.path.join(data_dir, fold)
        images, masks, types = load_pannuke_fold(fold_path)
        all_images.append(images)
        all_masks.append(masks)
        all_types.append(types)

    all_images = np.concatenate(all_images)
    all_masks = np.concatenate(all_masks)
    all_types = np.concatenate(all_types)

    return all_images, all_masks, all_types