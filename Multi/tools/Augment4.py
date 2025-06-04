import cv2
import os
import random
import numpy as np

def read_label_file(label_file_path):
    labels = []
    with open(label_file_path, 'r') as file:
        for line in file:
            labels.append(line.strip().split(' '))
    return labels

def apply_random_operation(image, operation, bbox_left, bbox_top, bbox_right, bbox_bottom):
    # Calculate width and height of the bounding box
    width = bbox_right - bbox_left
    height = bbox_bottom - bbox_top

    # Calculate the coordinates for the four equal parts
    half_width = max(1, width // 2)
    half_height = max(1, height // 2)

    regions = [
        (bbox_left, bbox_top, bbox_left + half_width, bbox_top + half_height),      # Top-left quarter
        (bbox_left + half_width, bbox_top, bbox_right, bbox_top + half_height),     # Top-right quarter
        (bbox_left, bbox_top + half_height, bbox_left + half_width, bbox_bottom),   # Bottom-left quarter
        (bbox_left + half_width, bbox_top + half_height, bbox_right, bbox_bottom)   # Bottom-right quarter
    ]

    # Randomly select one region to apply the operation
    selected_region = random.choice(regions)
    region_left, region_top, region_right, region_bottom = selected_region

    # Ensure selected region is valid
    if region_left < region_right and region_top < region_bottom:
        if operation == 'dropout':
            # Apply dropout (set pixels to black)
            image[region_top:region_bottom, region_left:region_right] = 0

        elif operation == 'noise':
            # Apply Gaussian noise
            noise = np.random.normal(0, 25, (region_bottom - region_top, region_right - region_left, 3)).astype(np.uint8)
            image[region_top:region_bottom, region_left:region_right] += noise
            image = np.clip(image, 0, 255)

        elif operation == 'blur':
            # Apply Gaussian blur
            blur_region = image[region_top:region_bottom, region_left:region_right]
            if blur_region.size > 0:  # Ensure blur_region is not empty
                blurred_region = cv2.GaussianBlur(blur_region, (15, 15), 0)
                image[region_top:region_bottom, region_left:region_right] = blurred_region

    return image

def draw_2d_bounding_boxes_with_random_operations(image, labels):
    for label in labels:
        bbox_left = int(float(label[4]))
        bbox_top = int(float(label[5]))
        bbox_right = int(float(label[6]))
        bbox_bottom = int(float(label[7]))

        # Randomly select an operation to apply
        #operations = ['dropout', 'noise', 'blur']
        operations = [ 'noise', 'blur']
        #operations = ['dropout']
        selected_operation = random.choice(operations)

        # Apply the selected operation to a random quarter of the bounding box
        image = apply_random_operation(image, selected_operation, bbox_left, bbox_top, bbox_right, bbox_bottom)

    return image

def process_images(image_dir, label_dir, output_dir, interval=10):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of all image files in the input directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    for i, image_file in enumerate(image_files):
        if i % interval == 0:
            # Construct full paths to the image and corresponding label file
            image_file_path = os.path.join(image_dir, image_file)
            label_file_path = os.path.join(label_dir, image_file.replace('.png', '.txt'))

            # Read the image
            image = cv2.imread(image_file_path)
            if image is None:
                print(f'Error reading image {image_file_path}')
                continue

            # Read the labels
            labels = read_label_file(label_file_path)

            # Apply random operations to 2D bounding boxes
            image_processed = draw_2d_bounding_boxes_with_random_operations(image, labels)

            # Modify the filename (change the first digit to '1')
            new_image_filename = '1' + image_file[1:]

            # Construct the output file path
            output_file_path = os.path.join(output_dir, new_image_filename)

            # Save the processed image
            cv2.imwrite(output_file_path, image_processed)

            print(f'Processed {image_file} and saved as {new_image_filename}')

# 使用相對路徑
image_directory = './datasets/coco/TrainImg'
label_directory = './datasets/coco/label'
output_directory = './datasets/coco/TrainImgAug_4*2_hfull'

# 處理圖片
process_images(image_directory, label_directory, output_directory, interval=2)
