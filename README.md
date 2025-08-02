# 🚀 DR-TTA: Dynamic and Robust Test-Time Adaptation

![Model Architecture](img/model.png)

**DR-TTA** is a **test-time adaptation method** that enhances segmentation performance under domain shifts. It employs a **dual-branch Teacher-Student architecture**, where:

- 🧑‍🏫 The **Teacher** provides pseudo-labels,
- 👨‍🎓 The **Student** adapts using augmented target samples.

✨ Key modules include:
- Momentum updates
- Adaptive Batch Normalization
- Robust feature alignment

---

## 🧠 Visual Comparison of Segmentation

Comparison of results on **BRATS-SSA** and **BRATS-SIM** datasets.

> **NoTTA** indicates performance *before* applying domain adaptation.

🎨 **Color Legend**:  
- WT = red + green + blue  
- TC = red + blue  
- ET = red

![Segmentation Comparison](img/VIS.png)

---

## ⚙️ 1. Environment Setup

Python version: `3.8`

```bash
conda create -n DR-TTA python=3.8.20
conda activate DR-TTA
pip install -r requirements.txt
```

---

## 📁 2. Data Preparation

- **BraTS 2024-SSA**  
Pre-processed multi-modal MRI scans (T1, T1Gd, T2, FLAIR).  
🔗 [Download from Synapse](https://www.synapse.org/Synapse:syn59059780)

- **BraTS 2024-SIM**  
Use the script to generate the synthetic dataset:

```bash
python sim_dataset_maker.py
```

---

## 🏋️ 3. Pre-training on Source Domain (BraTS 2024)

Run the following script to train the model in the source domain:

```bash
python train_source.py
```

---

## 🧪 4. Test-Time Adaptation on Target Domain (SSA / SIM)

To perform test-time adaptation, run:

```bash
python run_3d_upl.py
```

This script includes both adaptation training and testing.

---

## 📄 Citation

If you find this work helpful, please consider citing us:

```bibtex
% Add your citation here when the paper is published.
```

---

🧑‍💻 *For any issues or questions, feel free to open an issue or contact the authors.*
