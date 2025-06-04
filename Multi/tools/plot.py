import cv2
import matplotlib.pyplot as plt
import os
import random

def read_label_file(label_file_path):
    labels = []
    with open(label_file_path, 'r') as file:
        for line in file:
            labels.append(line.strip().split(' '))
    return labels

def draw_2d_bounding_boxes_with_dropout(image, labels):
    for label in labels:
        class_name = label[0]
        bbox_left = int(float(label[4]))
        bbox_top = int(float(label[5]))
        bbox_right = int(float(label[6]))
        bbox_bottom = int(float(label[7]))

        # Draw the original bounding box
        cv2.rectangle(image, (bbox_left, bbox_top), (bbox_right, bbox_bottom), (0, 255, 0), 2)
        cv2.putText(image, class_name, (bbox_left, bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate width and height of the bounding box
        width = bbox_right - bbox_left
        height = bbox_bottom - bbox_top

        # Calculate the coordinates for the four equal parts
        mid_x = bbox_left + width // 2
        mid_y = bbox_top + height // 2

        quarters = [
            (bbox_left, bbox_top, mid_x, mid_y),          # Top-left quarter
            (mid_x, bbox_top, bbox_right, mid_y),         # Top-right quarter
            (bbox_left, mid_y, mid_x, bbox_bottom),       # Bottom-left quarter
            (mid_x, mid_y, bbox_right, bbox_bottom)       # Bottom-right quarter
        ]

        # Randomly select one quarter to dropout
        dropout_quarter = random.choice(quarters)
        dropout_left, dropout_top, dropout_right, dropout_bottom = dropout_quarter

        # Apply dropout (set pixels to black)
        image[dropout_top:dropout_bottom, dropout_left:dropout_right] = 0

        # Draw the dividing lines for visualization
        cv2.line(image, (mid_x, bbox_top), (mid_x, bbox_bottom), (255, 0, 0), 1)
        cv2.line(image, (bbox_left, mid_y), (bbox_right, mid_y), (255, 0, 0), 1)

    return image

def display_image(image):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# 使用相對路徑
image_filename = './datasets/coco/TrainImg/000000.png' # 替換為實際圖片名稱
label_filename = './datasets/coco/label_1/000000.txt' # 替換為實際標註文件名稱

# 讀取圖片
image = cv2.imread(image_filename)

# 讀取標註
labels = read_label_file(label_filename)

# 繪製2D邊界框並隨機丟棄四分之一
image_with_boxes = draw_2d_bounding_boxes_with_dropout(image, labels)

output_filename = './datasets/output_with_boxes.png'
cv2.imwrite(output_filename, image_with_boxes)

print(f'Output image saved to {output_filename}')