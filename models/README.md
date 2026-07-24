\# Models



This folder contains the YOLOv8 model weights used for training and inference.



\## Models Used



| Model | Description |

|--------|-------------|

| yolov8n.pt | Nano model (Fastest, lightweight) |

| yolov8s.pt | Small model (Balanced speed and accuracy) |

| yolov8m.pt | Medium model (Higher accuracy) |



\## Download



The pretrained weights can be downloaded automatically by Ultralytics or manually from:



https://github.com/ultralytics/assets/releases



or



https://docs.ultralytics.com/models/yolov8/



\## Training



To train a custom model:



```bash

yolo detect train model=yolov8m.pt data=coco\_custom.yaml epochs=50 imgsz=640

```



\## Validation



```bash

yolo detect val model=runs/detect/train/weights/best.pt

```



\## Inference



```bash

python main.py

```



\## Note



Large model weights (.pt files) are generally excluded from GitHub repositories to keep the repository lightweight. They can be downloaded using the links above.

