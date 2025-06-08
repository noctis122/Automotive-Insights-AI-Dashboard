
# Automotive Insights AI Dashboard

A powerful AI-powered dashboard that analyzes car reviews to provide **sentiment analysis**, **toxicity detection**, **summarization**, **translation**, and **question answering** — helping automotive dealerships and manufacturers understand customer feedback at scale.

Built using:

* HuggingFace Transformers Pipelines
* Pandas & Matplotlib
* Gradio Interface
* Python 3.11

## 🎯 Project Goals

✅ Provide deep insights from customer car reviews
✅ Automatically detect positive/negative sentiment
✅ Identify potential toxic or biased reviews
✅ Summarize long reviews into key points
✅ Translate reviews for multilingual analysis
✅ Enable interactive visualization and easy reporting

---

## 📊 Features

* **Sentiment Analysis:** Classifies reviews as POSITIVE or NEGATIVE
* **Toxicity Detection:** Highlights potentially harmful or toxic content
* **Summarization:** Generates concise summaries of long reviews
* **Translation:** Converts English reviews into Spanish (extendable to other languages)
* **Question Answering:** Extracts key information about brand perception
* **PDF/Markdown Report Generation:** For business insights
* **Interactive Gradio Dashboard:** User-friendly web interface

---


---

## 📂 Project Structure

```
├── car_review_analysis.py          # Main analysis class
├── app.py                          # Gradio app interface
├── automotive_insights_report.md   # Generated report
├── car_reviews_sample.csv          # Sample dataset
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
└── car_review_analysis.log         # Log file
```

---

## 📥 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/automotive-insights-ai-dashboard.git
cd automotive-insights-ai-dashboard
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Gradio app

```bash
python app.py
```

### 4️⃣ Upload your CSV file

CSV format:

| Review                                    | Class    |
| ----------------------------------------- | -------- |
| The car is very comfortable and reliable. | POSITIVE |
| Poor fuel efficiency and bad support.     | NEGATIVE |
| ...                                       | ...      |

---

## 📈 Example CSV

A sample file `car_reviews_sample.csv` is provided to test the app.

---

## 📝 Models Used

| Task               | Model Name                                      |
| ------------------ | ----------------------------------------------- |
| Sentiment Analysis | distilbert-base-uncased-finetuned-sst-2-english |
| Translation        | Helsinki-NLP/opus-mt-en-es                      |
| Question Answering | deepset/minilm-uncased-squad2                   |
| Summarization      | facebook/bart-large-cnn                         |
| Toxicity Detection | unitary/toxic-bert                              |

---

## 🎓 Motivation

This project was created to:

* Explore the power of NLP in automotive customer feedback
* Provide businesses with actionable insights
* Improve AI skills with real-world use cases
* Build an interactive web-based AI app

---

## 🛠️ Possible Improvements

* Add support for more languages
* Enable multi-class sentiment detection (e.g. neutral, mixed)
* Use more advanced QA models (RAG)
* Deploy as a cloud service (Streamlit, HuggingFace Spaces, AWS)
* Add keyword extraction and topic modeling

---

## 🙌 Acknowledgements

Thanks to:

* [HuggingFace](https://huggingface.co/) for providing pre-trained NLP models
* [Gradio](https://gradio.app/) for easy app deployment
* [Matplotlib](https://matplotlib.org/) for visualization

---

## 🧑‍💻 Author

**Mouhib Farhat**
Student | AI & Data Science Enthusiast
IEEE CS Chapter Project Manager

Connect with me:

* [LinkedIn](https://www.linkedin.com/in/yourprofile)
* [GitHub](https://github.com/yourusername)

---

## 📜 License

This project is licensed under the MIT License — feel free to use and adapt!

---
