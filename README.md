# Armrest Height Classification App

This is a **Streamlit-based web application** that analyzes a **side-profile image** of a person working at a desk and classifies whether their **chair armrest height** is:

- **Optimal**
- **Too High**
- **Too Low**

It uses **image processing** and **pose landmark detection** to assess ergonomic posture and provides:
- An **annotated image** with detected points and armrest height indicators.
- A **JSON output** containing detection details.

---

## 🚀 Features
- Upload images in `.png`, `.jpg`, `.jpeg`, or `.webp` format.
- Real-time armrest height classification.
- Visualization of detected pose landmarks.
- Detailed JSON data for further analysis.

---

## 📦 Requirements

Make sure you have Python installed (**>=3.9** recommended).

Install dependencies from the included `requirements.txt`:

```bash
pip install -r requirements.txt
```


## How to Run
```bash
streamlit run streamlit_app.py
```
