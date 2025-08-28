#  Crop Disease Detection System

##  Project Overview
The **Crop Disease Detection System**  detects plant diseases from leaf images.  
It provides real-time predictions along with actionable agricultural advice to help farmers manage crop health effectively.

---

##  Task Objective
- Train a robust **classification model** using transfer learning.  
- Deploy the trained model as a **REST API** for real-time inference.  
- Enhance predictions with **image preprocessing** (brightness/contrast correction).  
- Provide farmers with **prevention and cure** recommendations.  

---

## Tech Stack
- **Python 3.10+**  
- **TensorFlow / Keras** → Model training & inference  
- **FastAPI + Uvicorn** → REST API deployment  
- **Pillow** → Image preprocessing  
- **rembg** → Background removal  
- **Matplotlib** → Training visualization  
- **JSON** → Class labels & logs  

---

##  System Workflow
1. **Data Collection & Preprocessing**
   - Dataset organized into labeled folders.
   - Augmentation applied: rotation, scaling, flipping.
   - Training/validation split: **80/20**.

2. **Model Training & Evaluation**
   - Transfer learning with **MobileNetV2**.
   - Trained for **15 epochs**.
   - Saved as:
     - Model → `dataset.h5`
     - Class names → `class_names.json`.

3. **API Development**
   - Routes:
     - `POST /predict` → Upload leaf image, get prediction + advice.
     - `GET /` → Health check.
   - Preprocessing:
     - Background removal
     - Brightness/contrast correction
     - Resize & normalize  
   - Logging:
     - Saves inputs + results for traceability.

4. **Deployment**
   - Served using **Uvicorn**.

---

## Dataset
- **Classes**:
  - Healthy
  - Rust
  - Powdery  
- **Training Distribution**:
  - Healthy → 430 images  
  - Rust → 430 images  
  - Powdery → 430 images  

---

## Results & Findings
- Validation accuracy: **>90%**.  
- Transfer learning reduced training time and improved performance.  
- Preprocessing improved robustness on **real phone images**.  
- Logging helps monitor predictions.  

---

## Challenges
- **Overfitting** → Model performed well on dataset but poorly on real-world images initially.  
- **Background noise** → Misclassifications occurred.  
- **rembg limitations** → Background removal sometimes failed.  
- **Large uploads** → Caused performance/memory issues.  

---

## Future Improvements
- Collect **more diverse field data** for training.  
- Improve **background removal pipeline**.  
- Optimize for **mobile deployment**.  
- Add **multi-disease classification** support.  

---


# Run FastAPI server
uvicorn main:app --reload
