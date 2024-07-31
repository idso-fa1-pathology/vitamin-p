# src/data/preprocessing.py

import tensorflow as tf
import numpy as np
import os

def load_pannuke_fold(fold_path):



    fold_number = os.path.basename(fold_path).split()[-1]  # Extract the fold number
    images = np.load(os.path.join(fold_path, 'images', f'fold{fold_number}', 'images.npy'))
    masks = np.load(os.path.join(fold_path, 'masks', f'fold{fold_number}', 'masks.npy'))
    types = np.load(os.path.join(fold_path, 'images', f'fold{fold_number}', 'types.npy'))
    
    # Ensure masks are 2D
    if masks.ndim == 1:
        masks = masks.reshape(-1, 256, 256)  # Assume 256x256 shape, adjust if necessary
    elif masks.ndim == 2:
        masks = masks.reshape(-1, 256, 256)  # Assume 256x256 shape, adjust if necessary
    elif masks.ndim > 3:
        masks = np.sum(masks, axis=-1) > 0  # Convert multi-class to binary
    
    # Ensure masks are float32 and in range [0, 1]
    masks = masks.astype(np.float32)
    
    print(f"Loaded images shape: {images.shape}")
    print(f"Loaded masks shape: {masks.shape}")
    print(f"Loaded types shape: {types.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Masks dtype: {masks.dtype}")
    print(f"Masks min and max: {np.min(masks)}, {np.max(masks)}")
            # Add these print statements
    print(f"Raw masks shape: {masks.shape}")
    print(f"Raw masks dtype: {masks.dtype}")
    print(f"Raw masks min and max: {np.min(masks)}, {np.max(masks)}")
    
    return images, masks, types

def preprocess_images(images):
    # Normalize images to [0, 1] range
    images = images.astype(np.float32) / 255.0
    return images

def create_tf_dataset(images, masks, batch_size):
    def augment(image, mask):
        # Ensure mask is 3D for flip operation
        mask = tf.expand_dims(mask, axis=-1)
        
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        # Remove the extra dimension from mask
        mask = tf.squeeze(mask, axis=-1)
        return image, mask

    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def split_data(images, masks, types, fold):
    total_samples = len(images)
    if total_samples < 3000:
        raise ValueError(f"Not enough samples to split. Total samples: {total_samples}")

    if fold == 1:
        train_end = int(0.7 * total_samples)
        val_end = int(0.85 * total_samples)
    elif fold == 2:
        train_start = int(0.15 * total_samples)
        train_end = int(0.85 * total_samples)
        val_end = total_samples
    elif fold == 3:
        train_start = int(0.3 * total_samples)
        val_start = int(0.85 * total_samples)
    else:
        raise ValueError("Invalid fold number. Choose 1, 2, or 3.")

    if fold == 1:
        train = (images[:train_end], masks[:train_end], types[:train_end])
        val = (images[train_end:val_end], masks[train_end:val_end], types[train_end:val_end])
        test = (images[val_end:], masks[val_end:], types[val_end:])
    elif fold == 2:
        train = (images[train_start:train_end], masks[train_start:train_end], types[train_start:train_end])
        val = (images[:train_start], masks[:train_start], types[:train_start])
        test = (images[train_end:], masks[train_end:], types[train_end:])
    else:  # fold == 3
        train = (images[train_start:], masks[train_start:], types[train_start:])
        val = (images[val_start:], masks[val_start:], types[val_start:])
        test = (images[:train_start], masks[:train_start], types[:train_start])

    print(f"Train set size: {len(train[0])}")
    print(f"Validation set size: {len(val[0])}")
    print(f"Test set size: {len(test[0])}")
        # Add these print statements at the end of the function
    print(f"Train set size: {len(train[0])}, mask min/max: {np.min(train[1])}/{np.max(train[1])}")
    print(f"Validation set size: {len(val[0])}, mask min/max: {np.min(val[1])}/{np.max(val[1])}")
    print(f"Test set size: {len(test[0])}, mask min/max: {np.min(test[1])}/{np.max(test[1])}")

    return train, val, test

def preprocess_pannuke_data(data_dir, fold, batch_size):
    images, masks, types = load_and_preprocess_pannuke(data_dir)
    train, val, test = split_data(images, masks, types, fold)

    train_dataset = create_tf_dataset(train[0], train[1], batch_size)
    val_dataset = create_tf_dataset(val[0], val[1], batch_size)
    test_dataset = create_tf_dataset(test[0], test[1], batch_size)

        # Add these print statements before creating the datasets
    print("Train masks min and max:", np.min(train[1]), np.max(train[1]))
    print("Validation masks min and max:", np.min(val[1]), np.max(val[1]))
    print("Test masks min and max:", np.min(test[1]), np.max(test[1]))


    return train_dataset, val_dataset, test_dataset, types

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

    # Preprocess images
    all_images = preprocess_images(all_images)

    return all_images, all_masks, all_types