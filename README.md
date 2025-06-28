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

## 🦙 Install Ollama (Required for Image/Combined Models)

**Note:** If you want to run the Image Model (`AutogradImageModel.py`) or Combined Model (`Combined.py`), you must install Ollama and the LLaVA model.

### Step 1: Install Ollama
- Visit [https://ollama.com/download](https://ollama.com/download) and download Ollama for your operating system
- Follow the installation instructions for your platform

### Step 2: Pull LLaVA Model
After installing Ollama, run the following command in your terminal:

```bash
ollama pull llava:latest
```
> **Note:** The LLaVA model is large (approximately 4–5 GB). Ensure you have sufficient disk space and a stable internet connection before downloading.
**Important:** The Text Model (`AutogradTextModel.py`) does not require Ollama and can be run without it.

---

You're now ready to go! 🚀
# Usage

## 🖼️ Run Image Model

```bash
python AutogradImageModel.py
```

## 📝 Run Text Model

```bash
python AutogradTextModel.py
```

## 🔀 Run Combined Model

```bash
python Combined.py
```

*After running any of the models, you'll get a local URL in the terminal like:*
* Running on local URL:  http://127.0.0.1:7861

*Go to this link in your browser to access the implementation.*

## Demo

**Demo currently unavailable.**


