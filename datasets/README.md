\# Dataset



This project uses the \*\*COCO 2017 (Common Objects in Context)\*\* dataset for training and evaluation.



\## Dataset Information



\- Dataset: COCO 2017

\- Class Used: Person

\- Annotation Format: COCO JSON

\- Converted Format: YOLO



\## Download



Download the dataset from:



https://cocodataset.org/#download



Required files:



\- train2017.zip

\- val2017.zip

\- annotations\_trainval2017.zip



\## Folder Structure



```

datasets/

└── coco2017/

&#x20;   ├── images/

&#x20;   │   ├── train2017/

&#x20;   │   └── val2017/

&#x20;   ├── labels/

&#x20;   │   ├── train2017/

&#x20;   │   └── val2017/

&#x20;   └── annotations/

```



\## Annotation Conversion



The script `convert\_coco\_to\_yolo.py` converts COCO annotations into YOLO label format by extracting only the \*\*person\*\* class.



Run:



```bash

python convert\_coco\_to\_yolo.py

```



\## Note



The COCO dataset is not included in this repository because of its large size.

