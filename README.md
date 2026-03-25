# 📊 StatLab Pro — Statistical Data Analysis Tool

A production-grade Streamlit application for advanced statistical analysis, EDA, hypothesis testing, and ML model insights.

---

## 🚀 Features

| Module | Capabilities |
|--------|-------------|
| **Data Input** | CSV/Excel upload, GitHub raw URL loading |
| **Preprocessing** | Missing value handling, type conversion, outlier detection, feature selection |
| **EDA** | Summary stats (skew/kurtosis), correlation heatmap, distribution plots, pair plots |
| **Statistical Tests** | t-test, ANOVA, Chi-square, Shapiro-Wilk, Pearson/Spearman, confidence intervals |
| **Visualization** | 10+ interactive Plotly chart types, dynamic filters, HTML download |
| **Model Insights** | Linear/logistic regression, feature importance, residual plots |
| **Export** | CSV/Excel dataset download, CSV/PDF statistical report |

---

## 🛠️ Local Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/statlab-pro.git
cd statlab-pro

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deploying on Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — StatLab Pro"
git remote add origin https://github.com/YOUR_USERNAME/statlab-pro.git
git branch -M main
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click **"Deploy!"**

Streamlit Cloud will automatically install packages from `requirements.txt`.

### Step 3 — Environment Variables (optional)

If your app uses API keys or secrets, add them in:
**Streamlit Cloud → App settings → Secrets**

```toml
# .streamlit/secrets.toml (local only, never commit this file)
GITHUB_TOKEN = "ghp_xxx"
```

---

## 📁 Project Structure

```
statlab-pro/
├── app.py              # Main Streamlit application
├── utils.py            # Statistical & ML utility functions
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

---

## 📦 .gitignore

```
venv/
__pycache__/
*.pyc
.env
.streamlit/secrets.toml
*.egg-info/
dist/
```

---

## 🧪 Loading Sample Data via GitHub URL

Use any raw GitHub CSV URL, for example:

```
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```

---

## 📝 License

MIT License — free to use and modify.
