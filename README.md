# Armrest Height Classification App

https://docs.google.com/document/d/1lau08V3qay5ETC-SD0nOiTa20RGRS7px_Nwa9Go0oJM/edit?usp=sharing

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
# App Screenshots
<img width="248" height="773" alt="image" src="https://github.com/user-attachments/assets/a6305b37-f8ea-4eb9-8c2c-57e649ea75b8" />
<img width="898" height="379" alt="image" src="https://github.com/user-attachments/assets/f7e3dc65-eb47-4726-b97c-3afcf50da2ae" />

