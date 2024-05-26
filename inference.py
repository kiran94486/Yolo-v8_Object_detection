import os
import argparse
import cv2
from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def infer_and_draw_boxes(model, image_path, output_path, color, thickness):
    image = cv2.imread(image_path)
    results = model(image)
    for result in results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox[:4])
            conf = bbox[4]
            class_id = int(bbox[5])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(image, f'{model.names[class_id]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(output_path, image)

def main(input_dir, output_dir, person_det_model_path, ppe_det_model_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    person_model = load_model(person_det_model_path)
    ppe_model = load_model(ppe_det_model_path)

    for image_file in os.listdir(input_dir):
        if image_file.endswith(('.jpg', '.png')):
            input_image_path = os.path.join(input_dir, image_file)
            output_image_path = os.path.join(output_dir, image_file)

            # Run person detection
            infer_and_draw_boxes(person_model, input_image_path, output_image_path, color=(0, 255, 0), thickness=2)
            
            # Run PPE detection on the detected persons (cropped images)
            person_results = person_model(cv2.imread(input_image_path))
            for result in person_results:
                for bbox in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cropped_image = cv2.imread(input_image_path)[y1:y2, x1:x2]
                    cropped_output_path = os.path.join(output_dir, f'cropped_{image_file}')
                    cv2.imwrite(cropped_output_path, cropped_image)
                    infer_and_draw_boxes(ppe_model, cropped_output_path, cropped_output_path, color=(255, 0, 0), thickness=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference using YOLOv8 models")
    parser.add_argument('input_dir', type=str, help="Path to the input directory containing images")
    parser.add_argument('output_dir', type=str, help="Path to the output directory to save results")
    parser.add_argument('person_det_model', type=str, help="Path to the person detection model")
    parser.add_argument('ppe_detection_model', type=str, help="Path to the PPE detection model")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)
