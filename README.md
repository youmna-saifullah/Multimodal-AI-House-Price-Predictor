# VisionValuate: Multimodal AI House Price Predictor

**Author:** Youmna Saifullah  
**Internship:** Developer Hub (Phase 2 Task)  
**Domain:** Deep Learning, Computer Vision, & Multimodal Fusion

---

## 📌 Project Statement
Traditional real estate models rely solely on tabular data (e.g., square footage, number of bedrooms). However, these models ignore a critical factor: **aesthetic appeal and condition.** A house with "3 bedrooms" might be a luxury renovation or a fixer-upper. 

**VisionValuate** uses a Multimodal Deep Learning approach to extract visual features from house images (Kitchen, Bathroom, Bedroom, and Living Room) using a **Convolutional Neural Network (CNN)** and combines them with structured metadata using a **Multi-Layer Perceptron (MLP)** to predict the final sale price with significantly higher nuance.

---

## 🏗️ Model Architecture
The project implements a **Functional API** architecture to handle two distinct data streams:

1.  **Tabular Branch (MLP):** Processes numerical data (Beds, Baths, Area) through Dense layers with ReLU activation.
2.  **Image Branch (CNN):** Processes 64x64 synthesized "house montages" through three stages of Convolutional and Pooling layers to identify visual quality.
3.  **Fusion Layer:** Concatenates the feature vectors from both branches into a single regression head to output a continuous price value.



---

## 📊 Data Source & Preprocessing
To maintain repository efficiency, the raw dataset is not included in this repository.

* **Dataset:** [Houses Dataset (Ahmed & Moustafa)](https://github.com/emanhamed/Houses-dataset)
* **Image Synthesis:** Four room images (Kitchen, Bed, Living, Din) are resized to 32x32 and tiled into a **64x64 montage** for each property ID.
* **Scaling:** Tabular data is normalized using `MinMaxScaler`, and targets are scaled by the maximum price to ensure stable model convergence.

---

## 🛠️ Technology Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (cv2), Pillow
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Deployment:** Gradio

---

## 🚀 Installation & Setup

### 1. Clone the Repo
```bash
git clone [https://github.com/youmna-saifullah/Multimodal-AI-House-Price-Predictor.git](https://github.com/youmna-saifullah/Multimodal-AI-House-Price-Predictor.git)
cd Multimodal-AI-House-Price-Predictor

```

### 2. Install Dependencies

```bash
pip install tensorflow opencv-python scikit-learn gradio pandas numpy

```

### 3. Usage

Run the main notebook or script to launch the **Gradio UI**. You will need the saved model and scaler files:

* `house_price_model.keras`
* `scaler.pkl`
* `y_max.txt` (for denormalizing prices)

---

## 🖥️ Deployment (Gradio UI)

The project features an interactive dashboard where users can input house statistics and upload room images to receive an instant AI-powered valuation.

---

## 📈 Future Improvements

* Implement **Data Augmentation** on the image branch to reduce overfitting.
* Incorporate **Transfer Learning** (e.g., ResNet50) for advanced visual feature extraction.
* Add **Zip Code One-Hot Encoding** to factor in location-based pricing.

---

*This project was developed during my internship at **Developer Hub** to demonstrate advanced Multimodal Deep Learning capabilities.*

```
