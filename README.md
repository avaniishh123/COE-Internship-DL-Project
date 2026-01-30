# Generalizable Polyp Detection & Segmentation Across Endoscopy Devices  
### Domain-Adaptive DeepLabV3 with Gradient Reversal (DANN)

Polyp segmentation models often **fail in real clinical settings** because endoscopy devices differ in  
**lighting**, **sensor calibration**, **resolution**, **color response**, and **optical distortion**.  
A model trained on one device typically **collapses** when evaluated on another.

This project introduces a **device-agnostic polyp segmentation framework** using  
**Domain-Adversarial Neural Networks (DANN)** with a **Gradient Reversal Layer (GRL)** on top of  
**DeepLabV3 (ResNet-50)**, enabling **robust cross-device generalization**.

> **Key idea:** learn **domain-invariant representations** while preserving high segmentation accuracy.

---

## 🚀 Key Contributions

- ✅ **Domain-Adaptive Segmentation (DANN + GRL)** for cross-device robustness  
- ✅ **DeepLabV3 (ResNet-50)** backbone for strong semantic representation  
- ✅ **Unlabeled target-domain adaptation** (ETIS-Larib)  
- ✅ **Multi-polyp handling** via copy–paste augmentation  
- ✅ **Instance-level visualization** using bounding boxes & watershed  
- ✅ **End-to-end reproducible pipeline** with Colab + Kaggle + Drive checkpoints  

---

## 🧠 Method Overview (Why This Generalizes)

### Problem
A standard segmentation model learns **device-specific cues** → poor performance on unseen devices.

### Solution
We use **Domain-Adversarial Training**:

- A **shared encoder** extracts features
- A **segmentation head** predicts masks (source domain only)
- A **domain classifier** predicts source vs target
- A **Gradient Reversal Layer (GRL)** forces the encoder to **confuse domains**

This pushes the encoder to learn **device-invariant features**.

### Objective Function

<img width="217" height="29" alt="image" src="https://github.com/user-attachments/assets/49a34df8-dc0a-4bfc-8143-15df64e122b0" />


Where:
- **Segmentation Loss** = BCEWithLogits + Dice  
- **Domain Loss** = Cross-Entropy  
- **λ (domain weight)** = 0.1  

---

## 🏗️ Architecture Explanation (Domain-Adaptive Polyp Segmentation)

Below is a **clean, professional, GitHub-README–ready explanation** written **strictly with respect to the architecture diagram** you shared.
This is the kind of explanation reviewers, interviewers, and recruiters expect.

You can paste this **directly under the Architecture section** in your README.

---

## 🏗️ Architecture Explanation (Domain-Adaptive Polyp Segmentation)

The proposed architecture follows a **Domain-Adversarial Neural Network (DANN)** framework built on top of **DeepLabV3**, enabling **robust polyp segmentation across different endoscopy devices**.

The model operates in **two phases**: **Training** and **Inference**, as illustrated in the diagram.

                                                              <img width="784" height="501" alt="Screenshot 2026-01-30 173241" src="https://github.com/user-attachments/assets/3d6099cd-6340-4005-8fd6-34ba9d3509b8" />


---

## 🔹 Training Phase

During training, the model jointly learns **segmentation accuracy** and **domain invariance** using both labeled and unlabeled data.

---

### 1️⃣ Source Domain (Labeled)

**Input:**

* Labeled polyp images
* Corresponding ground-truth segmentation masks

**Process:**

* Images are passed through a **shared encoder (DeepLabV3 backbone)**.
* The extracted features are fed into a **Segmentation Head**.
* The predicted mask is compared with the ground-truth mask.

**Loss:**

* **Segmentation Loss** = Binary Cross-Entropy + Dice Loss
* Ensures accurate pixel-level polyp segmentation.

---

### 2️⃣ Target Domain (Unlabeled)

**Input:**

* Unlabeled polyp images from a different endoscopy device (e.g., ETIS-Larib)

**Process:**

* Images pass through the **same shared encoder**.
* No segmentation loss is applied (no labels available).
* Features are instead sent to a **Domain Classifier**.

---

### 3️⃣ Shared Encoder (DeepLabV3)

* Acts as a **common feature extractor** for both source and target domains.
* Learns high-level semantic features (polyp shape, texture, boundaries).
* Initially captures domain-specific information, which must be removed.

---

### 4️⃣ Gradient Reversal Layer (GRL)

The **key component enabling domain adaptation**.

* During forward pass: acts as an identity layer.
* During backpropagation: **reverses gradients** coming from the domain classifier.

**Effect:**

* The encoder is penalized if domain-specific features are learned.
* Forces the encoder to learn **domain-invariant representations**.

---

### 5️⃣ Domain Classifier

* Receives features from the encoder via the GRL.
* Predicts whether features come from:

  * **Source domain**
  * **Target domain**

**Loss:**

* **Domain Classification Loss (Cross-Entropy)**

**Adversarial Objective:**

* Domain classifier tries to distinguish domains.
* Encoder tries to **confuse** the classifier.

---

### 6️⃣ Total Training Loss

<img width="395" height="195" alt="image" src="https://github.com/user-attachments/assets/8bd1109b-95ed-4257-a6c2-b2a055537224" />


This joint optimization ensures:

* High segmentation accuracy
* Strong cross-device generalization

---

## 🔹 Inference Phase

During inference, **domain adaptation components are removed**.

**Input:**

* Any endoscopy image (from any device)

**Process:**

* Image → Shared Encoder → Segmentation Head

**Output:**

* Predicted polyp segmentation mask

✔ No domain classifier
✔ No gradient reversal
✔ Fully device-agnostic inference



## 📦 Datasets

| Dataset | Role | Description |
|------|------|------------|
| **CVC-ClinicDB** | Source | Labeled training dataset |
| **Kvasir-SEG** | External | Independent validation |
| **ETIS-Larib** | Target | Unlabeled domain adaptation |

> ⚠️ Dataset licenses are respected.  
> Kaggle download + manual placement both supported.

---

## 📊 Quantitative Results (Threshold = 0.5)

### A) **Source Domain Performance — CVC**

| Metric | Macro (Mean per-image) | Micro (Global pixels) |
|------|------------------------|-----------------------|
| Dice | **0.9200** | **0.9420** |
| IoU | 0.8641 | 0.8903 |
| Precision | 0.9080 | 0.9279 |
| Recall | 0.9426 | 0.9565 |
| Specificity | 0.9928 | 0.9931 |
| F1 Score | 0.9249 | 0.9420 |
| F2 Score | 0.9368 | 0.9506 |

---

### B) **Target Domain Performance — ETIS (Unseen Device)**

| Metric | Macro (Mean per-image) | Micro (Global pixels) |
|------|------------------------|-----------------------|
| Dice | **0.8539** | **0.9114** |
| IoU | 0.7788 | 0.8372 |
| Precision | 0.8260 | 0.8675 |
| Recall | **0.9139** | **0.9600** |
| Specificity | 0.9929 | 0.9930 |
| F1 Score | 0.8740 | 0.9114 |
| F2 Score | **0.9045** | **0.9399** |

> **Key Insight:**  
> Despite a domain shift, the model maintains **high recall and Dice**, indicating  
> **strong generalization without performance collapse**.

---

## 📈 Cross-Dataset Generalization Summary

### Kvasir-SEG (Independent Dataset)

| Metric | Score |
|------|------|
| Dice | **0.8358** |
| IoU | 0.7298 |
| Precision | 0.8362 |
| Recall | 0.8198 |
| Accuracy | 0.9480 |
| FPS | **452+** |

### 📌 Main Takeaway
Compared to the source domain (CVC):
- ETIS and Kvasir show a **controlled drop**, not a collapse
- Confirms **device-agnostic segmentation capability**
- Demonstrates **real-world robustness**

---

## 🖼️ Visualization & Explainability

- 🔥 Probability heatmaps over input images  
- 🟩 Bounding boxes for each detected polyp  
- 🧠 Watershed-based instance separation  
- 📊 Per-image Dice/IoU reporting  

Supports:
- Single polyp
- Multiple polyps in same frame
- Challenging lighting conditions

---

## 🧪 Multi-Polyp Augmentation (Optional)

To prevent **single-polyp bias**, we introduce **copy–paste augmentation**:

- Extract polyp from one image
- Paste into another image
- Update mask via union

This significantly improves:
- Multi-polyp detection
- Instance robustness

---

## 🚀 Live Demo

- 🔗 **Hugging Face App:** https://huggingface.co/spaces/Cowkur/polyp-segmentation-dann  
- 🎥 **Demo Video:** https://drive.google.com/file/d/11YvPun0sMPqP4ehYS6HdhyMGqKpdKFG-/view  

Upload any endoscopic image and get:
- Segmentation mask
- Bounding boxes
- Detection verdict

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/generalizable-polyp-segmentation.git
cd generalizable-polyp-segmentation
pip install -r requirements.txt
