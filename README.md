# Fake Video Detection

This project detects deepfake videos using a Convolutional Neural Network (CNN) trained on frames extracted from real and fake videos in the **Deepfake Face Mask Dataset (DFFMD)**. It classifies videos as either **REAL** or **FAKE** based on multiple frames sampled per video.

---

## Project Structure:

- `config.py` – Contains dataset paths and parameters like `IMG_SIZE`, `FRAMES_PER_VIDEO`, and `LIMITED_VIDEOS`.
- `data_loader.py` – Loads real and fake videos, extracts frames, and constructs the dataset.
- `model_builder.py` – Defines a deep CNN for binary classification (real vs fake).
- `train.py` – Loads the dataset, trains the model, evaluates it, and saves the trained model.
- `predict.py` – Loads the trained model and predicts whether a video is real or fake by analyzing its frames.

---

## Installation:

Make sure you have Python 3.7+ and install the following dependencies:

```bash
pip install numpy opencv-python scikit-learn tensorflow json5
```

## Dataset Location:

| Data Type    | Path                       |
|--------------|----------------------------|
| Fake Videos  | `/data/Fake/Fake/`         |
| Real Videos  | `/data/Real/Real/`         |
| Metadata     | `data/DFFD_metadata.json`  |


## Model Architecture:

The model is a deep CNN with the following structure:

	•	4 blocks of:

	•	Conv2D → BatchNormalization → MaxPooling2D → Dropout

	•	Followed by:

	•	Flatten → Dense(512) → Dense(256) → Dense(1, sigmoid)


## Frame Extraction:

	•	Frames per video: 10 evenly spaced frames

	•	Image size: 128×128 pixels

	•	Videos used: 100 real and 100 fake (configurable)


## Training the Model:

To train the model:

```bash
python train.py
```

This will:

	•	Extract frames from videos

	•	Train the CNN for 5 epochs

	•	Evaluate validation accuracy and loss

	•	Save the trained model to saved_model/deepfake_detection_model.h5


## Inference: Predict a Video

```bash
from predict import predict_video

result = predict_video("path/to/video.mp4")
print("Prediction:", result)
```



## Evaluation:

During training, the model reports:

	•	Training and validation accuracy/loss

	•	Final validation performance

	•	Saves the model after training


## Configurable Parameters:

| Parameter         | Description                                      |
|-------------------|--------------------------------------------------|
| `IMG_SIZE`        | Target image size (default: 128)                 |
| `FRAMES_PER_VIDEO`| Frames extracted per video (default: 10)         |
| `LIMITED_VIDEOS`  | Max videos to load from each class (default: 100)|


## License:

This project is intended for educational and research purposes. Feel free to use, modify, and build upon this work with credit.