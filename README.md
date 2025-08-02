# 🧠 DR-TTA: Dynamic and Robust Test-Time Adaptation

We propose a test-time adaptation method called DR-TTA (Dynamic and Robust Test-Time Adaptation). This method employs a dual-branch teacher-student architecture, where the teacher provides pseudo-label supervision, and the student adapts to domain shifts through augmented target samples. Additionally, DR-TTA integrates momentum updates and adaptive Batch Normalization to enhance feature alignment and maintain source knowledge.
![image](img/model.png)



# 💡 Primary Contributions
Despite advances in SFUDA, most methods still face two main challenges. The first is catastrophic forgetting, where the model loses the ability to recognize crucial pathological features from the source domain during adaptation. The second involves limited and often poor-quality of target domain data, which leads to low-confidence pseudo-labels. These unreliable labels can degrade model performance, particularly when noise interference is severe. To address these issues, we propose the Dynamic and Robust Test-Time Adaptation (DR-TTA) framework, which offers innovations across three key areas:

1、Parameter Freezing and Momentum Updating:In this collaborative framework, convolutional weights are frozen to retain critical source domain knowledge, while sample-aware adaptive BatchNorm layers are updated to facilitate cross-domain feature calibration. The teacher model uses momentum updating to create pseudo-labels, thereby reducing distribution bias.

2、Dynamic Data Augmentation Optimization:To counteract MRI-specific domain shifts, a backpropagation-driven mechanism dynamically selects from 11 predefined augmentations by optimizing combination weights. This adaptive selection generates high-quality augmented samples that better match the target domain's distribution, thereby enhancing model adaptability.

3、Hybrid Loss Function and Sample Screening:To stabilize training affected by low-confidence pseudo-labels, a dynamic sample screening strategy is implemented. This confidence-aware and noise-resistant approach eliminates noisy samples and adaptively suppresses their gradient contributions based on reliability estimates, thus improving training robustness and convergence stability.


# ⚡ Visual Comparison
Visual comparison of segmentation results on the BRATS-SSA and BRATS-SIM datasets. NoTTA indicates results before the different domain adaptation methods. Color legend: WT = red + green + blue, TC = red + blue, ET = red.
![image](img/VIS.png)

Visual comparison of segmentation results in the ablation study conducted on the BRATS-SSA dataset. The visual results demonstrate that removing any individual component from the DR-TTA framework leads to degraded segmentation quality, with notable boundary artifacts and region misclassifications. Color legend: WT = red + green + blue, TC = red + blue, ET = red.
![image](img/VIS_ablation.png)

## 🔧 Environment Setup
Please prepare an environment with Python 3.8, and then use the command "pip install -r requirements.txt" for the dependencies:

```
conda create -n DR-TTA python=3.8.20
conda activate DR-TTA
pip install -r requirements.txt

```



## 📁 Data Preparation
- BraTS 2024-SSA:
Pre-processed MRI scans (including T1, T1Gd, T2, and FLAIR) from the BRATS-SSA dataset (https://www.synapse.org/Synapse:syn59059780) were utilized in this study.

- BraTS 2024-SIM:

```
python sim_dataset_maker.py

```


## 🏋️ Pre-train on Source Domain (BraTS 2024)
Run "train_source.py" to get a pre-trained weight:

```
python train_source.py

```


## 🧪 Test-Time Adaptation in Target Domain (SSA/SIM)
Run "run_3d_upl.py" to get the result in the target domain. It contains both the training and test processes:

```
python run_3d_upl.py

```



## 📝 Citation

```


```
