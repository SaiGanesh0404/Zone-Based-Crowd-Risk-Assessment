\# Zone-Based Crowd Risk Assessment using YOLOv8



A real-time computer vision system that detects people in video streams and assesses crowd density across user-defined zones using the YOLOv8 object detection model. The application classifies each zone into \*\*Safe\*\*, \*\*Moderate\*\*, or \*\*High Risk\*\* based on the number of detected people, making it useful for crowd monitoring in public spaces, events, transportation hubs, and smart city applications.



\---



\## Features



\- Real-time person detection using YOLOv8

\- User-defined rectangular and polygonal monitoring zones

\- Live crowd counting for each zone

\- Automatic risk classification

&#x20; - 🟢 Safe

&#x20; - 🟠 Moderate

&#x20; - 🔴 High Risk

\- Interactive GUI using OpenCV

\- Zone deletion functionality

\- Model performance evaluation

\- Confusion Matrix generation

\- ROC Curve generation

\- Support for multiple YOLOv8 model variants



\---



\## Project Structure



```

Zone-Based-Crowd-Risk-Assessment/

│

├── datasets/

├── models/

├── outputs/

├── runs/

├── videos/

│

├── main.py

├── make\_metrics.py

├── convert\_coco\_to\_yolo.py

├── coco\_custom.yaml

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## Technologies Used



\- Python

\- OpenCV

\- YOLOv8 (Ultralytics)

\- NumPy

\- Matplotlib

\- Pandas

\- Scikit-learn

\- TQDM



\---



\## Dataset



This project uses the \*\*COCO 2017 Dataset\*\* and extracts only the \*\*Person\*\* class for training and evaluation.



Dataset:

https://cocodataset.org/#download



\---



\## Installation



Clone the repository



```bash

git clone https://github.com/YOUR\_USERNAME/Zone-Based-Crowd-Risk-Assessment.git

```



Go to the project directory



```bash

cd Zone-Based-Crowd-Risk-Assessment

```



Install dependencies



```bash

pip install -r requirements.txt

```



\---



\## Running the Project



Place your input video inside the `videos` folder.



Run the application



```bash

python main.py

```



\---



\## Performance Metrics



| Metric | Value |

|---------|--------|

| Precision | 0.824 |

| Recall | 0.729 |

| mAP@50 | 0.814 |

| mAP@50-95 | 0.606 |

| ROC-AUC | 0.9914 |



\---



\## Output



The system provides:



\- Real-time person detection

\- Live crowd count

\- Zone-wise risk level

\- Confusion Matrix

\- ROC Curve

\- Detection visualizations



\---



\## Results



\### Confusion Matrix



!\[Confusion Matrix](Confusion\_matrix.png)



\### ROC Curve



!\[ROC Curve](roc.png)



\### Detection Output



!\[Detection Result](result.png)



\---



\## Future Improvements



\- Multi-camera crowd monitoring

\- Heatmap visualization

\- Crowd flow prediction

\- Web dashboard

\- Alert notifications

\- Cloud deployment

\- Person tracking with DeepSORT or ByteTrack



\---



\## Author



\*\*Damastapur Sai Ganesh\*\*



GitHub: https://github.com/SaiGanesh0404

\---



\## License



This project is intended for educational and research purposes.



