### 📄 `README.md` — Sign Language Detection using ASL Alphabet Dataset

```markdown
# 🧠 Sign Language Detection using ASL Alphabet Dataset

This project uses a Convolutional Neural Network (CNN) model trained on the **ASL Alphabet dataset** to detect American Sign Language (A–Z) hand gestures from images  input.

---

## 📥 Dataset Download

Download the ASL Alphabet dataset from Kaggle:

🔗 [ASL Alphabet Dataset - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))

After downloading, extract the dataset and place the folder in your project directory as:

```

/asl\_alphabet\_train/

````

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/Vidhi4401/sign-language-detection.git
cd sign-language-detection
````

2. Install all required dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ You may want to create a virtual environment before installing.

---

## 🧠 Step 1: Train the Model

Run the training script:

```bash
python preprocess_and_train.py
```

This will:

* Preprocess the ASL dataset
* Train a CNN model on the images
* Save the trained model to disk (e.g., `asl_model.h5`)

> 🕒 Note: Training may take **2-3 hours** depending on your hardware.

---

## 🎯 Step 2: Detect Sign Language

Once training is complete, use the detection script to classify hand gestures:

```bash
python detect_real_time.py
```

The script will:

* Load the trained model
* Access your webcam or test image
* Display real-time predictions for detected hand signs

---

## 📁 Project Structure

```
sign-language-detection/
├── train.py                # Model training script
├── detect.py               # Detection using webcam
├── requirements.txt        # Required Python packages
├── asl_alphabet_train/     # ASL training dataset (from Kaggle)
├── model/asl_model.h5      # Trained model (after training)
```

---

## 🌐 Future Enhancements

* Support for Indian or other regional sign languages
* Integration with voice/text translation
* Web-based UI using Streamlit or Flask

---
