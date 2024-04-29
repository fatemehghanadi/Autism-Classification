# Autism-Classification
Wavelet Based fMRI Analysis for Autism Spectrum Disorder Detection using Feature Selection and Ridge Classifier

This repository presents a machine learning (ML) based framework designed to enhance ASD diagnosis using fMRI data. Leveraging techniques such as wavelet transform, Tangent Pearson (TP) embedding, Principal Component Analysis (PCA), Analysis of Variance (ANOVA) feature selection, and Maximum Independence Domain Adaptation (MIDA) algorithm, the framework extracts meaningful features from fMRI signals and aligns them for effective classification.

The workflow begins with wavelet transform to extract frequency levels from Blood Oxygen Level-dependent signals. Subsequently, TP embedding, PCA, and ANOVA are employed for feature reduction and selection. To address domain shift induced by different fMRI scanning types, MIDA is applied to align feature representations, ensuring maximal independence while preserving classification-relevant information.

The framework achieves state-of-the-art performance, with an Area Under the Curve (AUC) metric of 79.01% and an accuracy rate of 72.47%.

![Ovaral Pipeline](pipeline.png)

### Installation

To install the dependencies required for this project, run the following command:

```bash
pip install -r requirements.txt

## Acknowledgements

This project is built upon the code from [
fMRI-site-adaptation]() by [Mwiza Kunda]([link_to_original_author](https://github.com/kundaMwiza/fMRI-site-adaptation)). I utilized their MIDA and Tangent Pearson Embedding and made modifications to suit the requirements of my project.
