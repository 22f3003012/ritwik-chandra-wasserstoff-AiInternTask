

# DocFind AI – Intelligent Document Assistant

**DocFind AI** is an intelligent document processing assistant that allows users to chat with PDFs and images using advanced AI models. It combines OCR and LLM-powered insights to extract, index, and answer user queries over documents with ease and precision.

Hosted App: [Live on Hugging Face Spaces](https://huggingface.co/spaces/Ritwik1607/DocFindAI)

---
| Upload Section | Chat Section | Results Section |
|----------------|--------------|-----------------|
| ![Upload](assets/Screenshot%202025-06-13%20053911.jpg) | ![Chat](assets/Screenshot%202025-06-13%20053941.jpg) | ![Results](assets/Screenshot%202025-06-13%20054003.jpg) |


---

## 🚀 Features

- Upload multiple **PDFs and images** (PNG, JPG)
- Perform **OCR (Optical Character Recognition)** on uploaded files
- Extract **text, metadata, and context**
- Query documents using **LLMs** (powered by LLaMA-3 via LangChain)
- View **AI-powered chat responses**
- Get **citations** including document name, page number, paragraph, and extract
- Thematic summary generation
- Deployed with a beautiful custom UI (Gradio + CSS)

---

## 🧠 Tech Stack

- **Frontend**: Gradio (customized with CSS)
- **Backend**: Python, LangChain, PyMuPDF, EasyOCR, ChromaDB, FAISS
- **Model**: LLaMA-3-70B (via Hugging Face APIs)
- **Deployment**: Hugging Face Spaces
- **Version Control**: Git & GitHub

---

## 📂 Folder Structure

```

DocFindAI/
├── app.py                # Main Gradio app logic
├── assets/               # Screenshots & visual assets
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
└── .env.template         # Environment variable placeholder

````

---


## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ritwik-chandra/wasserstoff/AiInternTask.git
cd AiInternTask
````

### 2. Create virtual environment & activate

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API keys

* Rename `.env.template` to `.env`
* Add your Hugging Face API key and any others required.

### 5. Run locally

```bash
python app.py
```

---

## 📬 Submission Details

* GitHub Repo: [ritwik-chandra/wasserstoff/AiInternTask](https://github.com/ritwik-chandra/wasserstoff/AiInternTask)
* Hosted Link: [DocFind AI on Hugging Face](https://huggingface.co/spaces/Ritwik1607/DocFindAI)
* Report: Included in the repository as `DocFind_Report.pdf`
* Video Demo: Recorded and attached separately (if voice explanation is not included, please mention in email)

---



---

## 👨‍💻 Developed by

**Ritwik Chandra**
AI/ML Developer | GDSC Lead | B.Tech CSE (AIML)
[LinkedIn](https://linkedin.com/in/ritwik1607) • [GitHub](https://github.com/ritwik-chandra)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

---

```
