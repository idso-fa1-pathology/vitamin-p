import tensorflow as tf
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

def preprocess_image(image, target_size=(256, 256)):
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

def load_and_preprocess_he_data(data_dir):
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    images = []
    masks = []
    
    for image_path in image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = preprocess_image(image)
        
        mask_path = image_path.replace("images", "masks")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = preprocess_image(mask)
        
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def load_and_preprocess_mif_data(data_dir):
    # Similar to H&E data loading, but adapted for mIF data format
    pass

def split_data(images, masks, val_split=0.1, test_split=0.1):
    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=test_split, random_state=42
    )
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_images, train_masks, test_size=val_split / (1 - test_split), random_state=42
    )
    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)

def main():
    he_data_dir = "data/raw/H&E"
    mif_data_dir = "data/raw/mIF"
    
    he_images, he_masks = load_and_preprocess_he_data(he_data_dir)
    mif_images, mif_masks = load_and_preprocess_mif_data(mif_data_dir)
    
    he_data = split_data(he_images, he_masks)
    mif_data = split_data(mif_images, mif_masks)
    
    # Save processed data
    for data_type, data in zip(["H&E", "mIF"], [he_data, mif_data]):
        for split, (images, masks) in zip(["train", "val", "test"], data):
            np.save(f"data/processed/{data_type}/{split}_images.npy", images)
            np.save(f"data/processed/{data_type}/{split}_masks.npy", masks)

if __name__ == "__main__":
    main()
