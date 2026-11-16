# Bangla-Handwriting-Recognition_ASSignment4
# Bangla Handwritten Character Recognition System

**Skills for Industry Competitiveness and Innovation Program (SICIP)**  
**BRAC University**

## Assignment 04

**Prepared by:** Mrittika Devi Mati  
**Student ID:** 1000129281

## 1. Project Overview

This project implements a Bangla handwritten character recognition system using the BanglaLekha-Isolated dataset.

The system uses a Convolutional Neural Network (CNN) to classify handwritten Bangla characters into 84 classes. A Streamlit interface provides a drawing area for user input.

## 2. Dataset

Dataset: [BanglaLekha-Isolated — Mendeley Data](https://data.mendeley.com/datasets/hf6sf8zrkc/2)

- Total images: 166,105
- Number of classes: 84
- Image dimensions: 64 × 64
- Image format: Grayscale
- Training samples: 132,884
- Validation samples: 33,221

## 3. Model Architecture

The CNN contains:

- Three convolutional layers with 32, 64, and 128 filters.
- Max-pooling layers.
- A flattening layer.
- A dense layer with 256 neurons.
- A dropout layer with a rate of 0.4.
- An 84-class softmax output layer.

The model was trained using the Adam optimizer and sparse categorical cross-entropy loss.

## 4. Experiment Tracking

MLflow was used to track two experiments with different learning rates.

| Experiment | Learning rate | Validation accuracy |
|---|---:|---:|
| Experiment 1 | 0.001 | 87.54% |
| Experiment 2 | 0.0005 | 87.17% |

The first experiment achieved the higher validation accuracy.

## 5. Application

The Streamlit interface allows users to draw handwritten Bangla input. The application uses the available recognition components to generate predictions.

## 6. How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Train the model:

```bash
python train.py
```

Build the Docker image:

```bash
docker build -t bangla-ocr-app:0.1 .
```

Run the Docker container:

```bash
docker run -p 8501:8501 bangla-ocr-app:0.1
```

Start the MLflow interface:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

## 7. Limitations

The current CNN is trained on isolated handwritten character images. It classifies individual characters rather than complete handwritten words  almost perfectly.
Prediction Accuracy: The system does not always identify handwritten input correctly. Some predictions may differ from the actual character or word, particularly when handwriting is unclear or characters have similar visual patterns.

Handwriting Variations: Differences in writing styles
That's why we added ocr.

OCR Integration: The project also integrates Orus OCR (V0.1-alpha) to support handwritten text recognition. It processes handwritten input and attempts to generate readable Bangla text.Orus OCR V0.1-alpha was integrated as an additional recognition component. It uses an OCR pipeline to process handwritten text and produce predicted text output. This provides an alternative to classifying each isolated character independently.




Despite these limitations, combining CNN-based character classification with OCR provides a foundation for Bangla handwritten text recognition. The system can produce predictions close to the intended input in some cases, but further improvements in word-level recognition, OCR processing, and handwriting variation handling are needed.
## 8. Conclusion
This project demonstrates a Bangla handwritten recognition pipeline combining CNN-based character classification, OCR integration, MLflow experiment tracking, and a Streamlit interface. While the current system provides a foundation for handwritten Bangla recognition, improving prediction reliability and whole-word recognition remains an important direction for future development.
