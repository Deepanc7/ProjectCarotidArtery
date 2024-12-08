# **Carotid Artery**

## **Overview**  
This project is a machine learning-based system aimed at analyzing carotid artery images to assist in diagnosing and evaluating medical conditions. It focuses on preprocessing, model training, and prediction, providing an end-to-end solution for carotid artery analysis.

---

## **Features**  
- Image preprocessing and boundary detection for carotid artery regions.  
- Model training with dropout layers for improved generalization.  
- Hyperparameter configuration for tuning the model.  
- Prediction and evaluation tools for analyzing new data.  

---

## **Tech Stack**  
- **Programming Language**: Python  
- **Libraries/Frameworks**: TensorFlow, NumPy, OpenCV, and Matplotlib  
- **File Handling**: Uses `.txt` and `.py` files for configurations and operations  

---

## **Repository Structure**  
- **`preprocess.py`**: Scripts for image preprocessing and mask generation.  
- **`train.py`**: Handles the training of the machine learning model.  
- **`predict.py`**: Used for making predictions on new data.  
- **`model.py`**: Defines the neural network architecture.  
- **`Images/`**: Directory containing sample carotid artery images.  
- **`Masks/`**: Directory with masks for image boundary identification.  

---

## **Installation and Usage**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Deepanc7/ProjectCarotidArtery.git
2. Navigate to the project directory:
  ```bash
cd ProjectCarotidArtery  
3. Install the required dependencies:

4. Run the preprocessing script:
```bash
python preprocess.py  
5. Train the model:
```bash
python train.py  
6. Run the prediction script for new images:
```bash
python predict.py  
