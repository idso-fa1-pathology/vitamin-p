#!/bin/bash

# Source directory containing MOS folders
SOURCE_DIR="/rsrch9/home/plm/idso_fa1_pathology/TIER1/yasin-multi-modality/HE&mIF_ROIs"

# Destination directory
DEST_DIR="/rsrch9/home/plm/idso_fa1_pathology/TIER2/yasin-vitaminp/Pathology_val_dsp"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Create a temporary Python script
cat > /tmp/crop_image.py << 'EOF'
import sys
from PIL import Image

def crop_image(input_path, output_path, size=512):
    try:
        img = Image.open(input_path)
        # Crop 512x512 from top-left corner
        cropped = img.crop((0, 0, size, size))
        cropped.save(output_path)
        return True
    except Exception as e:
        print(f"Error cropping {input_path}: {e}")
        return False

if __name__ == "__main__":
    crop_image(sys.argv[1], sys.argv[2])
EOF

# Loop through MOS folders (MOS001 to MOS080)
for mos_folder in "$SOURCE_DIR"/MOS*; do
    # Get the folder name (e.g., MOS001)
    folder_name=$(basename "$mos_folder")
    
    # Create subfolder in destination
    dest_subfolder="$DEST_DIR/$folder_name"
    mkdir -p "$dest_subfolder"
    
    # Find the first H&E image in this folder
    he_image=$(find "$mos_folder" -name "*_HE.png" -type f | head -n 1)
    
    if [ -z "$he_image" ]; then
        echo "No H&E image found in $folder_name, skipping..."
        continue
    fi
    
    # Get the base name and construct the mIF image path
    base_name=$(basename "$he_image" "_HE.png")
    mif_image="$mos_folder/${base_name}_mIF.png"
    
    # Check if mIF image exists
    if [ ! -f "$mif_image" ]; then
        echo "Paired mIF image not found for $he_image, skipping..."
        continue
    fi
    
    echo "Processing $folder_name: $base_name"
    
    # Crop 512x512 from H&E image
    python3 /tmp/crop_image.py "$he_image" "$dest_subfolder/${base_name}_HE_crop.png"
    
    # Crop 512x512 from mIF image
    python3 /tmp/crop_image.py "$mif_image" "$dest_subfolder/${base_name}_mIF_crop.png"
    
    echo "Saved cropped images to $dest_subfolder"
done

# Clean up temporary Python script
rm /tmp/crop_image.py

echo "Processing complete!"