# 🧬 AI CT-2 Replacement Assignment  
## BioPython + Sequence Analysis + Machine Learning for Deafness Variants

---

## 📌 1. Assignment Overview

This project implements a complete bioinformatics and AI pipeline for analyzing deafness-associated genetic variants.

The system performs:

- FASTA sequence parsing using BioPython  
- Variant parsing from VCF-style TSV file  
- Genomic region mapping using BED file  
- Mutation analysis (SNP / Insertion / Deletion)  
- Transition vs Transversion detection  
- Pattern analysis (A, C, G, T frequencies + GC content + k-mers)  
- Machine Learning classification (Pathogenic vs Benign)  
- Train / Validation / Test split  
- Log Loss (Cross-Entropy) computation  

This satisfies the AI CT-2 replacement requirements:
- Sequence Analysis  
- Pattern Analysis  
- Training  
- Testing  
- Validation  
- Loss Function  

---

## 📂 2. Dataset Description

### 🔹 flanking_sequences.fasta
- Contains ~300 bp DNA sequences  
- Each header represents a genomic region  
  Example: `chr1:6425049-6425349`  
- Parsed using BioPython  

### 🔹 deafness_vcf.tsv
- ~42,000 genetic variants  
- Contains:
  - Chromosome  
  - Position  
  - Reference allele (ref)  
  - Alternate allele (alt)  
  - Clinical significance  
  - Molecular consequence  
  - Gene name  
  - Disease name  

### 🔹 deafness_flanking.bed
- Maps each variant to its flanking genomic region  
- Uses 0-based coordinates  
- Used to link variants → FASTA sequence  

---

## 🧬 3. Sequence Analysis

For each flanking sequence:

- Sequence length  
- Frequency of A, C, G, T  
- GC content  
- 3-mer (k-mer) frequency distribution  

GC content formula:

GC = (G + C) / total_length  

This provides nucleotide composition pattern analysis.

---

## 🔬 4. Mutation Analysis

For each variant:

- SNP (Single Nucleotide Polymorphism)  
- Insertion  
- Deletion  
- Length change calculation  
- Transition detection (A↔G, C↔T)  
- Transversion detection (all other substitutions)  

This evaluates biological mutation patterns.

---

## 📊 5. Pattern Analysis

The following features are engineered:

### Sequence Features
- A%, C%, G%, T%  
- GC content  
- Sequence length  

### Mutation Features
- SNP flag  
- Insertion flag  
- Deletion flag  
- Length change  
- Transition flag  
- Transversion flag  

### k-mer Features
- All possible 3-mer combinations (AAA, AAC, …, TTT)  
- Normalized frequency vector  

These features form the AI model input vector.

---

## 🤖 6. Machine Learning Model

### 🎯 Task

Binary classification:

- 1 → Pathogenic  
- 0 → Benign  

Uncertain or conflicting variants are excluded.

---

### 📊 Data Split

- 70% Training  
- 15% Validation  
- 15% Test  

Training Set → Learns parameters  
Validation Set → Performance tuning  
Test Set → Final unbiased evaluation  

---

### 🧠 Model Used

**Logistic Regression**

Reasons:
- Suitable for binary classification  
- Interpretable  
- Stable baseline model  
- Appropriate for structured biological features  

---

## 📉 7. Loss Function

The model uses **Log Loss (Cross-Entropy Loss)**:

Loss = - ( y * log(p) + (1 - y) * log(1 - p) )

Where:
- y = true label  
- p = predicted probability  

Lower loss indicates better probability calibration.

---

## 📈 8. Evaluation Metrics

The model reports:

- Accuracy  
- ROC-AUC  
- Confusion Matrix  
- Log Loss  

Typical performance:

- Accuracy ≈ 0.75–0.78  
- ROC-AUC ≈ 0.74–0.76  

(Exact values depend on dataset split.)

---

## 📁 9. Output Files

After running the script, an `outputs/` folder is generated:

- mutation_summary.csv  
- sequence_summary.csv  
- ml_metrics.txt  
- test_predictions.csv  

These files demonstrate:
- Mutation distribution  
- Pattern statistics  
- Model performance  
- Prediction probabilities  

---

## 🏗 10. Project Structure

```
AI_CT2_Replacement/
│
├── FASTA.py
├── flanking_sequences.fasta
├── deafness_vcf.tsv
├── deafness_flanking.bed
│
├── outputs/
│   ├── mutation_summary.csv
│   ├── sequence_summary.csv
│   ├── ml_metrics.txt
│   └── test_predictions.csv
│
└── README.md
```

---

## 🚀 11. How To Run

### Step 1: Install Dependencies

```
pip install biopython pandas numpy scikit-learn
```

### Step 2: Run the Program

```
python FASTA.py
```

### Step 3: Check Outputs

All result files will appear inside the `outputs/` directory.

---

## 🎓 12. Concepts Demonstrated

- BioPython FASTA parsing  
- Genomic coordinate mapping (0-based vs 1-based systems)  
- Variant type analysis  
- Transition vs transversion logic  
- Feature engineering from biological sequences  
- Supervised machine learning pipeline  
- Proper dataset splitting methodology  
- Cross-Entropy loss interpretation  
- Model evaluation metrics  

---

## 📌 13. Learning Outcome

This assignment demonstrates:

- Integration of bioinformatics data formats  
- Understanding of mutation classification  
- Implementation of pattern analysis  
- Application of machine learning to genomics  
- Interpretation of AI loss functions  
- Proper experimental evaluation methodology  

---

## 👨‍💻 Author

AI CT-2 Replacement Assignment  
Department of Computer Science & Engineering  

---

## 📄 Academic Note

This repository was developed as part of an academic coursework submission for AI CT-2 Replacement.
