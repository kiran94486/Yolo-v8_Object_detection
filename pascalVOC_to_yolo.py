import os
import argparse
import xml.etree.ElementTree as ET

def convert_pascal_voc_to_yolo(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    class_mapping = {}
    with open(os.path.join(input_dir, 'classes.txt')) as f:
        classes = f.read().strip().split()
        class_mapping = {cls: idx for idx, cls in enumerate(classes)}

    annotations_dir = os.path.join(input_dir, 'annotations')
    output_annotations_dir = os.path.join(output_dir, 'labels')
    if not os.path.exists(output_annotations_dir):
        os.makedirs(output_annotations_dir)

    for annotation_file in os.listdir(annotations_dir):
        if annotation_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotations_dir, annotation_file))
            root = tree.getroot()
            image_filename = root.find('filename').text.replace('.jpg', '.txt')
            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)
            with open(os.path.join(output_annotations_dir, image_filename), 'w') as yolo_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in class_mapping:
                        continue
                    class_id = class_mapping[class_name]
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    x_center = (xmin + xmax) / 2.0 / image_width
                    y_center = (ymin + ymax) / 2.0 / image_height
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height
                    yolo_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLO format")
    parser.add_argument('input_dir', type=str, help="Path to the input directory containing annotations and classes.txt")
    parser.add_argument('output_dir', type=str, help="Path to the output directory to save YOLO formatted annotations")
    args = parser.parse_args()
    convert_pascal_voc_to_yolo(args.input_dir, args.output_dir)
