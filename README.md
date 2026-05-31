[README (1).md](https://github.com/user-attachments/files/28445543/README.1.md)
#  Skin Cancer Detection - projet-IA

A web application that detects skin cancer from images using deep learning, built with Flask and TensorFlow/Keras.

##  Description

This project  uses a VGG16 deep learning model trained to classify skin lesion images and detect potential signs of skin cancer. The application provides a simple web interface where users can upload an image and receive a prediction.

---

##  Features

- Upload a skin lesion image for analysis
- AI-powered prediction using a VGG16 model
- Patient management dashboard
- User authentication (login system)
- Results stored in a MySQL database (via XAMPP)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python / Flask |
| AI Model | TensorFlow / Keras (VGG16) |
| Database | MySQL (XAMPP / phpMyAdmin) |
| Frontend | HTML / CSS (Jinja2 templates) |

---

## 📁 Project Structure

```
projet-IA/
├── app.py                       # Main Flask application
├── model/
│   └── vgg16_skin_cancer.h5    # Trained model (not included, see below)
├── static/
│   └── style.css                # Stylesheet
├── templates/
│   ├── login.html               # Login page
│   ├── dashboard.html           # Main dashboard
│   ├── patients.html            # Patient list
│   ├── predict.html             # Image upload & prediction
│   └── result.html              # Prediction result
├── .gitignore
└── README.md
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/Skandar65gh-bit/projet-IA.git
cd projet-IA
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies

```bash
pip install flask tensorflow keras mysql-connector-python pillow
```

### 4. Configure the database (XAMPP)

1. Download and install **XAMPP** from https://www.apachefriends.org
2. Open the **XAMPP Control Panel** and start **Apache** and **MySQL**
3. Open **phpMyAdmin** at http://localhost/phpmyadmin
4. Create a new database (e.g. `skin_cancer_db`)
5. Import the SQL schema if provided, or let the app create the tables on first run
6. Update the database connection settings in `app.py`:

```python
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",        # default XAMPP password is empty
    database="skin_cancer_db"
)
```

### 5. Download the model

The model file (`vgg16_skin_cancer.h5`) is not included in this repository due to its size (129 MB).  
Download it and place it in the `model/` folder.

### 6. Run the application

```bash
python app.py
```

The app will be available at **http://127.0.0.1:5000**

---

##  Usage

1. Open the app in your browser
2. Log in with your credentials
3. Go to the **Predict** page
4. Upload a skin lesion image
5. Get the AI prediction result instantly

---

##  Author

**Skandar** — [@Skandar65gh-bit](https://github.com/Skandar65gh-bit)

---

