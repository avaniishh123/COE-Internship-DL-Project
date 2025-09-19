<img width="325" height="182" alt="image" src="https://github.com/user-attachments/assets/8e68227d-22e1-492e-bd66-0b48f944d00d" /># Generalizable Polyp Detection and Segmentation Across Endoscopy Devices

## 📌 Problem

Polyp segmentation models often fail to generalize across different **endoscopy devices** due to variations in lighting, resolution, color calibration, and imaging angles. This limits their clinical applicability.

## 🎯 Research Goal

Develop a **device-agnostic, domain-adaptive segmentation model** for robust polyp detection and segmentation.
Our approach combines:

* ✅ **Adversarial Domain Adaptation (DANN-UNet)**
* ✅ **Data Augmentation & Domain Randomization**
* ✅ **Self-Training with Pseudo Labels**
* ✅ (Optional) **Meta-Learning for Cross-Domain Adaptation**

## 📂 Datasets Used

* **Kvasir-SEG**
* **CVC-ClinicDB**
* **ETIS-Larib Polyp DB**

## 🏗️ Model Architecture

* **Base:** Extended U-Net
* **Adaptation:** Gradient Reversal Layer (GRL) with Domain Classifier
* **Loss Functions:**

  * Segmentation → BCE + Dice Loss
  * Domain Adaptation → BCEWithLogitsLoss

## 📊 Results (Kvasir-SEG Example)

| Metric     | Score      |
| ---------- | ---------- |
| Dice Score | **0.8358** |
| IoU        | **0.7298** |
| Precision  | **0.8362** |
| Recall     | **0.8198** |
| Accuracy   | **0.9480** |
| FPS        | **452+**   |

## 🚀 Installation & Usage

### 1. Clone Repository

```bash
git clone https://github.com/your-username/generalizable-polyp-segmentation.git
cd generalizable-polyp-segmentation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

Download datasets and place them in the `data/` folder:

* [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
* [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
* [ETIS-Larib Polyp DB](https://polyp.grand-challenge.org/EtisLarib/)


### 4. Training

```bash
python train_dann_unet.py --epochs 20 --batch_size 4
```

### 5. Evaluation

```bash
python evaluate.py --weights dann_unet_polyp.pth
```

### 6. Visualization

```bash
python visualize_predictions.py
```

### 7. Architecture diagram 

🏗️ Model Architecture

🔎 Explanation

The architecture is based on Domain-Adversarial Neural Network (DANN) + U-Net for polyp segmentation.

🔹 Inputs

Source Domain (Labeled): Provides labeled polyp images for supervised segmentation training.

Target Domain (Unlabeled): Supplies unlabeled images from a different device/domain, used for domain adaptation.

Target Sample (Optional): Can be used for domain confusion learning or visualization.

🔹 Model Core (Center)

Encoder: Extracts feature maps from both source and target domain images.

Shared Bottleneck: High-level representation shared across both tasks (segmentation + domain classification).

Gradient Reversal Layer (GRL): Ensures that the encoder learns domain-invariant features by flipping gradients during backpropagation.

Domain Classifier: Learns to distinguish source vs. target domain features. Works adversarially with GRL.

Segmentation Head: Decodes features to produce a pixel-wise segmentation mask.

🔹 Outputs

Predicted Mask (Target): Model’s segmentation prediction for polyp regions on target images.

Ground Truth Mask (Source): Actual segmentation mask used for supervised training (only available for source domain).


<img width="325" height="182" alt="image" src="https://github.com/user-attachments/assets/7c9c3cec-9f86-47d4-81a7-ac0ae506975a" />

### Great question 👍 For the **screenshots** you pasted (bounding boxes + masks results), you should create a **Results & Visualization** section in your README. This will highlight **qualitative performance** of your model.

Here’s how you can write it 👇

---

### 📸 Results & Visualization

To validate the performance of our **Generalizable Polyp Segmentation Model**, we tested on **unseen endoscopic images** across different domains.

### 🔹 What the Screenshots Show

1. **Input Endoscopic Image:** Raw image uploaded by the user.
2. **Detection Results:** Model detects polyp regions and draws **bounding boxes**.
3. **Segmentation Output:** Binary segmentation masks showing exact polyp boundaries.
4. **Detection Verdict:** Displays number of detected regions with confidence scores.

### 🔹 Example Outputs

* **Case 1:** Large polyp with clear boundary detected with high confidence.
  ![Result 1]![WhatsApp Image 2025-07-21 at 14 38 28_2b90f82b](https://github.com/user-attachments/assets/86a407c8-8286-4053-97fb-24a67762c698)


* **Case 2:** Small polyp localized with bounding box and segmented mask.
  ![Result 2]![WhatsApp Image 2025-07-21 at 14 38 43_791d7d7d](https://github.com/user-attachments/assets/59cd3746-8c2c-487a-bc6d-9134841317e0)


* **Case 3:** Polyp detected in challenging lighting, segmentation mask successfully outlines region.
  ![Result 3]![WhatsApp Image 2025-07-21 at 14 39 01_c0e370f4](https://github.com/user-attachments/assets/dc952f66-33d4-46d5-8b2e-f652eabac753)


* **Case 4:** Multiple polyps detected in same frame, bounding boxes drawn around each region.
  ![Result 4]![WhatsApp Image 2025-07-21 at 14 42 36_f0864fb2](https://github.com/user-attachments/assets/985a4b9b-6f33-4a78-a84f-bf0fb479fafa)

---


## 🔮 Future Work

* Incorporate **semi-supervised learning** with unlabeled data
* Explore **meta-learning for device-invariant representations**
* Clinical validation across **multi-center datasets**

## 👨‍💻 Authors

* Avanish Cowkur
* Abhishek Dhaladhuli
* Srinivasa Pagadala
