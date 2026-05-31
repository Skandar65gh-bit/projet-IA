from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import numpy as np
import os
from PIL import Image as PILImage
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2


app = Flask(__name__)
app.secret_key = 'skin_cancer_secret'

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = tf.keras.models.load_model('model/vgg16_skin_cancer.h5')

def get_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='skin_cancer_db'
    )

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = PILImage.open(img_path).resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    result = 'Malignant' if pred > 0.5 else 'Benign'
    return result, float(pred)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        db.close()
        if user:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Identifiants incorrects', 'danger')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) as total FROM patients")
    total = cursor.fetchone()['total']
    cursor.execute("SELECT COUNT(*) as malignant FROM patients WHERE result='Malignant'")
    malignant = cursor.fetchone()['malignant']
    cursor.execute("SELECT COUNT(*) as benign FROM patients WHERE result='Benign'")
    benign = cursor.fetchone()['benign']
    db.close()
    return render_template('dashboard.html', total=total, malignant=malignant, benign=benign)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        try:
            name = request.form['name']
            age = request.form['age']
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                result, prob = predict_image(filepath)
                gradcam_path = generate_gradcam(filepath, model)
                db = get_db()
                cursor = db.cursor()
                cursor.execute(
                    "INSERT INTO patients (name, age, result, probability, image_path) VALUES (%s, %s, %s, %s, %s)",
                    (name, age, result, prob, filepath)
                )
                db.commit()
                db.close()
                return render_template('result.html', result=result, prob=round(prob * 100, 2), img=filepath, gradcam=gradcam_path)
            else:
                flash('Format image invalide.', 'danger')
        except Exception as e:
            flash(f'Erreur: {str(e)}', 'danger')
    return render_template('predict.html')

@app.route('/patients')
def patients():
    if 'user' not in session:
        return redirect(url_for('login'))
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
    patients_list = cursor.fetchall()
    db.close()
    return render_template('patients.html', patients=patients_list)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

from flask import jsonify

@app.route('/patients_data')
def patients_data():
    if 'user' not in session:
        return jsonify({'total': 0, 'malignant': 0, 'benign': 0})
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) as total FROM patients")
    total = cursor.fetchone()['total']
    cursor.execute("SELECT COUNT(*) as malignant FROM patients WHERE result='Malignant'")
    malignant = cursor.fetchone()['malignant']
    cursor.execute("SELECT COUNT(*) as benign FROM patients WHERE result='Benign'")
    benign = cursor.fetchone()['benign']
    db.close()
    return jsonify({'total': total, 'malignant': malignant, 'benign': benign})
#plusssssssssssssss
def generate_gradcam(img_path, model):
    # Charger et préparer l'image
    img = PILImage.open(img_path).resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Modèle qui retourne la dernière conv + la sortie
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('block5_conv3').output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    # Calcul des gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Appliquer la heatmap sur l'image originale
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    # Sauvegarder
    gradcam_filename = 'gradcam_' + os.path.basename(img_path)
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    cv2.imwrite(gradcam_path, superimposed)

    return gradcam_path

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)