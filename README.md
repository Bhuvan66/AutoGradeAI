# AutoGrade AI

**AutoGrade AI** is a Machine Learning-based automated grading system that leverages ensemble learning and state-of-the-art language and vision models to evaluate student answers with high accuracy and fairness.

## 🔍 Overview

AutoGrade AI aims to automate the evaluation of student responses by combining textual similarity metrics with image understanding for diagram-based answers. It uses a **Random Forest** model to intelligently weigh and combine multiple semantic similarity features.

## 🧠 Key Components

### 1. **Textual Answer Evaluation**
- Utilizes advanced language models to compute the similarity between student answers and reference answers:
  - **BERT**: Base transformer embeddings for semantic context.
  - **SBERT (Sentence-BERT)**: For more accurate sentence-level embeddings.
  - **KeyBERT**: To extract and compare keyphrases.

These similarity scores are used as features in a **Random Forest** classifier/regressor to predict scores.

### 2. **Diagram Evaluation**
- For questions involving diagrams or visual responses:
  - Uses **LLaVA** (Large Language and Vision Assistant) to generate descriptions of both the reference and student diagrams.
  - Applies the same BERT/SBERT/KeyBERT pipeline to evaluate semantic similarity between the generated descriptions.

## ✅ Features
- Supports both textual and diagrammatic answer evaluation.
- Model-agnostic pipeline for easy extension to other similarity metrics or vision models.
- Enables scalable, consistent, and explainable grading.

## 🚀 Technologies Used
- Python, scikit-learn (RandomForest)
- Hugging Face Transformers (BERT, SBERT)
- KeyBERT for keyphrase extraction
- LLaVA for multimodal image-to-text processing

## 📈 Applications
- Educational platforms
- Online examination systems
- Intelligent tutoring systems
# Installation

## 📦 Clone the Repository

```bash
git clone https://github.com/Bhuvan66/AutoGradeAI.git
cd AutoGradeAI
```

## 🛠️ Create Virtual Environment

### For macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
```

### For Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

---

You're now ready to go! 🚀
# Usage

## 🖼️ Run Image Model

```bash
python3 AutogradImageModel.py
```

## 📝 Run Text Model

```bash
python3 AutogradTextModel.py
```

## 🔀 Run Combined Model

```bash
python3 Combined.py
```

## Demo

Insert gif or link to demo

