import argparse
from pathlib import Path
import random
import cv2
import numpy as np
from ultralytics import YOLO

WIDTH = 2
FONT_SCALE = 1

# parse arguments
parser = argparse.ArgumentParser(description='Waste object detection with YOLO using camera, image, or video')
parser.add_argument(
    '--model-path',
    type=str,
    required=True,
    help='Path to the pretrained YOLO model (.pt file).'
)
parser.add_argument(
    '--input',
    type=str,
    required=True,
    help='Input source: webcam (e.g., "0") or image file path.'
)
parser.add_argument(
    '--conf',
    type=float,
    required=False,
    default=0.5,
    help='Confidence threshold for detecting objects (0 to 1).'
)
args = parser.parse_args()

# validate model
model_path = Path(args.model_path).absolute()
if not model_path.exists() or model_path.suffix != '.pt':
    raise ValueError(f'Invalid model path: {args.model_path}')

# Validate input source
input_source = None
if args.input.isdigit() and int(args.input) >= 0:
    input_source = int(args.input)  # webcam index
elif Path(args.input).exists():
    input_source = str(Path(args.input).absolute())  # image/video file path
else:
    raise ValueError(f'Invalid input source: {args.input}')

conf_threshold = args.conf

# load model from given weight
model = YOLO(str(model_path))

# class color mapping
class_colors = dict()
def get_class_color(class_id):
    global class_colors
    if class_id not in class_colors:
        # random color if not exists
        class_colors[class_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    return class_colors[class_id]

# process input source
if isinstance(input_source, int):  # webcam
    cap = cv2.VideoCapture(input_source)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # yolo inference
        results = model(frame, conf=conf_threshold)

        for result in results:
            for box in result.boxes:
                # get bounding box coordinates and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.int32)
                cls = box.cls.cpu().numpy().astype(np.int32)[0]
                cls_name = result.names[cls]

                color = get_class_color(cls)

                # draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, WIDTH)
                cv2.putText(frame, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, WIDTH)

        # show result
        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    # check if the input is a video or an image
    file_extension = Path(input_source).suffix.lower()
    is_video = file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

    if is_video:  # process video
        cap = cv2.VideoCapture(input_source)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # yolo inference
            results = model(frame, conf=conf_threshold)

            for result in results:
                for box in result.boxes:
                    # get bounding box coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.int32)
                    cls = box.cls.cpu().numpy().astype(np.int32)[0]
                    cls_name = result.names[cls]

                    color = get_class_color(cls)

                    # draw box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, WIDTH)
                    cv2.putText(frame, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, WIDTH)

            # show result
            cv2.imshow('output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    else:  # process image
        image = cv2.imread(input_source)

        # yolo inference
        results = model(image, conf=conf_threshold)

        for result in results:
            for box in result.boxes:
                # get bounding box coordinates and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.int32)
                cls = box.cls.cpu().numpy().astype(np.int32)[0]
                cls_name = result.names[cls]

                color = get_class_color(cls)

                # draw box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), color, WIDTH)
                cv2.putText(image, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, WIDTH)

        # Show result
        cv2.imshow('output', image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
