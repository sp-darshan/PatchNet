import albumentations as A
import cv2
import os
import numpy as np
import random
import pandas as pd
from utils.utils import read_cfg

# Augmentation pipeline (only geometrical transformations)
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.7),
])

def augment_images(df, input_dir, output_dir, target_count):
    os.makedirs(output_dir, exist_ok=True)
    augmented_data = []

    for class_name, count in df['dx'].value_counts().items():
        if count >= target_count:
            continue  # Skip already balanced classes

        print(f"Augmenting class '{class_name}': {count} -> {target_count}")

        class_images = df[df['dx'] == class_name]['image_id'].tolist()
        num_images = len(class_images)

        while num_images < target_count:
            img_name = random.choice(class_images)
            img_path = os.path.join(input_dir, img_name + ".jpg")

            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ ERROR: Unable to read image - {img_path}")
                continue  # Skip this image

            augmented = augment(image=img)['image']

            aug_name = f"aug_{num_images}_{img_name}.jpg"
            aug_path = os.path.join(output_dir, aug_name)

            cv2.imwrite(aug_path, augmented)
            augmented_data.append([aug_name.strip(".jpg"), class_name])

            num_images += 1

    return augmented_data

cfg = read_cfg(cfg_file='D:\\skin\\config\\config.yaml')
df_train = pd.read_csv(cfg['train_set'])

input_dir = cfg['img_dir']
output_dir = cfg['img_dir']

target_count = 1350 

# Augment dataset to balance minority classes
augmented_data = augment_images(df_train, input_dir, output_dir, target_count)

# Convert augmented data to DataFrame
aug_df = pd.DataFrame(augmented_data, columns=['image_id', 'dx'])

# Ensure all columns exist in aug_df
for col in df_train.columns:
    if col not in aug_df.columns:
        aug_df[col] = None  

# Append augmented data
df_train = pd.concat([df_train, aug_df], ignore_index=True)

# **Undersample Majority Classes**
final_df = pd.DataFrame()
for class_name, count in df_train['dx'].value_counts().items():
    if count > target_count:
        # Randomly sample `target_count` images
        sampled_df = df_train[df_train['dx'] == class_name].sample(n=target_count, random_state=42)
    else:
        sampled_df = df_train[df_train['dx'] == class_name]
    
    final_df = pd.concat([final_df, sampled_df])

# Save updated dataset
final_df.to_csv("balanced_train.csv", index=False)

print("✅ Final balanced dataset saved!")
