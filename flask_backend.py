from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import sqlite3
import datetime
from ultralytics import YOLO
import cv2
import numpy as np
import hashlib
import qrcode

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure the upload and processed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the trained YOLO model
MODEL_WEIGHTS_PATH = "weights/best.pt"
pose_model = YOLO(MODEL_WEIGHTS_PATH)

# Initialize SQLite database
DATABASE = 'dogs.db'
conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS dogs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        keypoints TEXT NOT NULL,
        note TEXT,
        last_seen TIMESTAMP
    )
''')

# Add missing columns if they do not exist
try:
    cursor.execute("ALTER TABLE dogs ADD COLUMN note TEXT")
except sqlite3.OperationalError:
    pass

try:
    cursor.execute("ALTER TABLE dogs ADD COLUMN last_seen TIMESTAMP")
except sqlite3.OperationalError:
    pass

conn.commit()
conn.close()

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_keypoints(image, keypoints):
    """Draw keypoints on the image."""
    for kp_set in keypoints:
        for kp in kp_set:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

def generate_dog_id(name):
    """Generate a unique ID for the dog based on its name."""
    hash_object = hashlib.sha256(name.encode())
    return hash_object.hexdigest()

def generate_qr_code(data, filename):
    """Generate a QR code from data and save it as an image."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']
        dog_name = request.form.get('dog_name')
        note = request.form.get('note')

        # If no file or name is provided
        if file.filename == '' or not dog_name:
            return "No selected file or dog name provided", 400

        if file and allowed_file(file.filename):
            # Save the file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load the image and perform prediction
            image = cv2.imread(filepath)
            results = pose_model.predict(image, conf=0.3, iou=0.55)[0].cpu()

            if not len(results.keypoints.xy):
                return "No keypoints detected in the image.", 200

            # Draw keypoints on the image
            keypoints = results.keypoints.xy.numpy()  # Extract keypoint coordinates
            processed_image = draw_keypoints(image, keypoints)

            # Generate a unique ID for the dog based on its name
            dog_id = generate_dog_id(dog_name)

            # Save processed image
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, processed_image)

            # Store dog information in the database
            keypoints_str = np.array2string(keypoints.flatten(), separator=',')
            last_seen = datetime.datetime.now()
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()

            # Check if the dog ID already exists
            cursor.execute("SELECT * FROM dogs WHERE id = ?", (dog_id,))
            existing_dog = cursor.fetchone()

            if existing_dog:
                # Update existing record
                cursor.execute("UPDATE dogs SET keypoints = ?, note = ?, last_seen = ? WHERE id = ?", (keypoints_str, note, last_seen, dog_id))
            else:
                # Insert new record
                cursor.execute("INSERT INTO dogs (id, name, keypoints, note, last_seen) VALUES (?, ?, ?, ?, ?)", (dog_id, dog_name, keypoints_str, note, last_seen))

            conn.commit()

            # Fetch all records from the database
            cursor.execute("SELECT * FROM dogs")
            dogs = cursor.fetchall()
            conn.close()

            # Generate a QR code for the dog ID
            qr_code_filename = f"qr_{filename}.png"
            qr_code_filepath = os.path.join(app.config['PROCESSED_FOLDER'], qr_code_filename)
            generate_qr_code(dog_id, qr_code_filepath)

            # Return HTML with the processed image, QR code, and database records
            dogs_table = "<table border='1'><tr><th>ID</th><th>Name</th><th>Keypoints</th><th>Note</th><th>Last Seen</th></tr>"
            for dog in dogs:
                dogs_table += f"<tr><td>{dog[0]}</td><td>{dog[1]}</td><td>{dog[2]}</td><td>{dog[3]}</td><td>{dog[4]}</td></tr>"
            dogs_table += "</table>"

            additional_info = ""
            if existing_dog:
                additional_info = f"<p><strong>Note:</strong> {existing_dog[3]}</p><p><strong>Last Seen:</strong> {existing_dog[4]}</p>"
            print(dogs_table)
            return f'''
                <!doctype html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Where's My Dog - Image Processed</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            background-color: #f0f0f0;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            flex-direction: column;
                            min-height: 100vh;
                            margin: 0;
                        }}
                        h1 {{
                            color: #333;
                        }}
                        .container {{
                            background: white;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
                            text-align: center;
                        }}
                        .image-container {{
                            display: flex;
                            justify-content: space-around;
                            align-items: center;
                            gap: 20px;
                            margin-top: 20px;
                        }}
                        img {{
                            max-width: 100%;
                            height: auto;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }}
                        a {{
                            display: inline-block;
                            margin-top: 20px;
                            padding: 10px 20px;
                            text-decoration: none;
                            background-color: #007BFF;
                            color: white;
                            border-radius: 5px;
                        }}
                        a:hover {{
                            background-color: #0056b3;
                        }}
                        footer {{
                            margin-top: 40px;
                            color: #777;
                            font-size: 0.9em;
                        }}
                        table {{
                            margin-top: 20px;
                            border-collapse: collapse;
                            width: 100%;
                        }}
                        th, td {{
                            padding: 10px;
                            text-align: left;
                            border-bottom: 1px solid #ddd;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Where's My Dog - Dog's characters and QR Code</h1>
                        <div class="image-container">
                            <img src="/processed/{processed_filename}" alt="Processed Image">
                            <img src="/processed/{qr_code_filename}" alt="QR Code">
                        </div>
                        {additional_info}
                        <br><br>
                        <a href="/">Find Another Dog</a>
                    </div>
                    <footer>Group Project 10</footer>
                </body>
                </html>
            '''

    # Render the upload form
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Where's My Dog - Find your dog by photo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                min-height: 100vh;
                margin: 0;
            }
            h1 {
                color: #333;
            }
            .container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            input[type="file"] {
                margin: 20px 0;
            }
            input[type="text"] {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ccc;
                width: calc(100% - 22px);
            }
            input[type="submit"] {
                padding: 10px 20px;
                border: none;
                background-color: #007BFF;
                color: white;
                border-radius: 5px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #0056b3;
            }
            footer {
                margin-top: 40px;
                color: #777;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Where's My Dog - Upload photo</h1>
            <form method="post" enctype="multipart/form-data">
                <input type="text" name="dog_name" placeholder="Enter Dog's Name" required>
                <input type="file" name="file" accept="image/*" required>
                <br>
                <input type="text" name="note" placeholder="Where is it?">
                <br>
                <input type="submit" value="Find My Dog">
            </form>
        </div>
        <footer>Group Project 10</footer>
    </body>
    </html>
    '''


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

def main():
    app.run(host="0.0.0.0", port=1428, debug=True)

if __name__ == "__main__":
    main()
