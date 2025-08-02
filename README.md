# 🎬 IMDB Visual Analyzer

This is a simple and interactive web app that lets you explore IMDB movie data in a beautiful and easy-to-use way.  
You can search, filter, and visualize movie trends, and even get **future suggestions for directors** on which genres they should explore next.

It’s like a **movie data dashboard – but smarter**.  

---

## ✨ What You Can Do with It

- **Create an Account & Login** – So your session is personal to you.
- **Explore Movies** – See top-rated films, highest revenue earners, best directors, movies by genre/year, and more.
- **Interactive Charts** – Beautiful charts built with Plotly that you can hover and interact with.
- **PDF Reports** – Export the table and chart you’re looking at into a neat PDF file.
- **Future Analysis for Directors** – Choose a director and find out:
  - What genres they’ve worked on most.
  - How they compare to the overall industry.
  - Which genres are trending upward.
  - Which genres they should try next.

---

## 🛠 How It Works

- The app uses **Streamlit** to create the web interface.
- Movies are stored in a local **SQLite database** (`imdb_data.db`).  
  If it’s empty, the app will load them from the included **`imdb.csv`** file.
- Queries are run **directly on the database**, and results are shown in a table or chart.
- You can choose from a list of ready-made queries like **“Top 10 Rated Movies”** or **“Revenue by Genre.”**
- If you like what you see, click **Export to PDF** to download your results.

---

## 📂 What You Need

- Python **3.8+**
- `streamlit` – for the app interface  
- `pandas` – for handling the movie data  
- `plotly` – for interactive charts  
- `fpdf` – for PDF generation  
- `kaleido` – to export Plotly charts as images for the PDF  

---

## 🚀 How to Run

1️⃣ **Install the requirements**
```bash
pip install -r requirements.txt
