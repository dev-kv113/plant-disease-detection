# Plant Disease Detector (PyTorch)

This project trains a Convolutional Neural Network (CNN) using [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) to predict plant diseases from leaf images.

---

## 0. Install Prerequisites

To run this project locally, make sure you have these installed:

### Install Visual Studio Code
Download and install [Visual Studio Code](https://code.visualstudio.com/).

### Install Git
Download and install [Git](https://git-scm.com/downloads) for version control.

### Install Python
Install **Python 3.8 – 3.13** from the official site:  
[https://www.python.org/downloads/](https://www.python.org/downloads/)  
During installation:
- ✔ **Check “Add Python to PATH”**.
- Let it install `pip` (it’s included by default in Python installers).

To verify:
```bash
python --version
pip --version
````

If you see Python and pip versions printed, you’re good to go.

---
## 1. Dataset (Download and Extract)

**Note:** The dataset is too large to include in this repository.  
You need to download it manually before training.

**Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

**Steps to set up:**
1. Go to the link above and download the dataset (`plantvillage.zip` or similar).
   - You need a free Kaggle account to download.
2. Once downloaded, place the zip file in your project folder.
3. Extract it (right-click → Extract All…) in the same folder as your scripts.

You should end up with a structure like this:

```
plant-disease-detector/       
├─ PlantVillage/
│  ├─ Tomato___Early_blight/
│  ├─ Tomato___Late_blight/
│  └─ ...
├─ train_pytorch.py
├─ predict_pytorch.py
├─ requirements.txt
└─ README.md
```

Each class (disease) is a subfolder inside `PlantVillage/`, containing its own images.

## 2. Set Up Environment

Open the folder in **VS Code**.

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```
torch
torchvision
Pillow
```

---

## 3. Train the Model

Run the training script:

```bash
python train_pytorch.py
```

What it does:

* Loads all images from `PlantVillage/`
* Applies basic image transforms (resize to 128×128, convert to tensors)
* Splits dataset into 80% train / 20% validation
* Builds a small CNN
* Trains for 3 epochs
* Saves the trained model to `saved_model/plant_disease_model.pth`

Output:

```
Epoch 1/3, Loss: ...
Epoch 2/3, Loss: ...
Epoch 3/3, Loss: ...
✅ Model saved!
```

---

## 4. Predict a Leaf Image

After training, test with an image:

```bash
python predict_pytorch.py <path_to_leaf_image>
```

Example:

```bash
python predict_pytorch.py test.JPG
```

Output:

```
✅ Predicted: Pepper__bell___Bacterial_spot (confidence 0.95)
```

What it does:

* Loads the trained model
* Preprocesses the image (resize, normalize)
* Outputs the predicted class and confidence

---

## 5. Notes

* You can test with any image from `PlantVillage` to confirm predictions.
* There is already a test.JPG in the files for you to try out using the above command.
* You can also try other leaf images after training.
* To speed up training for a quick demo, reduce `EPOCHS = 1` in `train_pytorch.py`.
* For higher accuracy, increase `EPOCHS` (try 10 or 15).

---

## 6. Commands Summary

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train_pytorch.py

# Predict
python predict_pytorch.py test.JPG
```

---

### What this code does:

* **train\_pytorch.py:** Builds and trains a CNN model on your dataset.
* **predict\_pytorch.py:** Loads the trained model and predicts a single image.

---

