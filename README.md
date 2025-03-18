# CSE-144-Final-Project

**Title:** Swin Transformer for Image Classification
**Authors:** Suhas Oruganti, Tejas Polu, Ashwin Nagarajan 
**Emails:** sorugant@ucsc.edu, tpolu@ucsc.edu, asnagara@ucsc.edu
**Date:** 03/21/2025  

---  

## **1. Introduction**  
This report details the experimental setup, methodology, and results of using a **Swin Transformer** for image classification. We leverage advanced training techniques such as **MixUp, CutMix, and Test-Time Augmentation (TTA)** to improve model robustness and accuracy. The model is fine-tuned on a **custom dataset** and evaluated on a held-out validation set.  

## **2. Experimental Setup**  
### **2.1 Hardware & Software Requirements**  
- **Hardware:** Ran on a CPU for original testing, but GPU recommended
- **Software:**  
  - Python 3.8+  
  - PyTorch  
  - torchvision, tqdm, numpy, pandas, PIL  

### **2.2 Dataset Structure**  
- **Training Data:** Located in `train/train/` (subdirectories representing class labels: `0, 1, ..., 99`).  
- **Test Data:** Located in `test/test/` (images named `0.jpg, 1.jpg, ...`).  

## **3. Instructions to Run the Code**  
### **3.1 Installation & Setup**  
```bash  
pip install torch torchvision timm numpy pandas tqdm PIL  
```  
### **3.2 Running Training**  
```bash  
python train_swin.py  
```  
- The best model will be saved as `best_model.pth`.  

### **3.3 Running Inference on Test Data**  
```bash  
python infer.py  
```  
- Generates `submission.csv` with predictions.  

## **4. Experimental Techniques**  
### **4.1 Data Augmentation**  
- **Training Augmentations:** AutoAugment, Random Cropping, Color Jittering, Rotation, Random Erasing.  
- **Test-Time Augmentation (TTA):** Multiple transformed inferences per image, averaging results.  

### **4.2 MixUp & CutMix**  
- **Epochs 1-5:** Uses MixUp (blends two images).  
- **Epochs 6-10:** Uses CutMix (patch-based augmentation).  

### **4.3 Gradual Unfreezing**  
- **Epoch 4:** Unfreezes `layers.1`.  
- **Epoch 7:** Unfreezes `layers.0`.  

## **5. Results**  
| Metric | Value |  
|--------|--------|  
| Best Kaggle Validation Accuracy | **0.76%** |
| Train Accuracy | **0.81%** |

## **7 Model Weights**
Download final model weights from Google Drive: https://drive.google.com/drive/folders/17KVxn88uSz62oCHhaD4s8e6GemAA9tGU?usp=sharing

## **6. Conclusion**  
This experiment demonstrates how the **Swin Transformer** achieves high accuracy for image classification. Further improvements can be explored by optimizing hyperparameters and dataset augmentation strategies.  

