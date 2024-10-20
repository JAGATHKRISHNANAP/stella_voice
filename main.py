from flask import Flask, request, jsonify, session, render_template
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json
from speaker_reg_app import (
    recognize_user, record_audio, register_user, 
    load_model, init_db, extract_voice_embedding, 
    is_user_already_registered
)

app = Flask(__name__)
trainfilepath = "C:/Users/hp/Downloads/Speaker_App/train.wav"
testfilepath = "C:/Users/hp/Downloads/Speaker_App/test.wav"

@app.route('/')
def index():
    return render_template('index.html')  # Serve home page if needed.

@app.route('/register')
def register_page():
    return render_template('register.html')  # Serve the register page.

@app.route('/recognize')
def recognize_page():
    return render_template('recognize.html')  # Serve the recognize page.

@app.route('/register', methods=['POST'])
def handle_register():
    name = request.form.get('name')
    audio_file = request.files.get('audio')

    if name and audio_file:
        trainfilepath = "C:/Users/hp/Downloads/Speaker_App/train.wav"
        audio_file.save(trainfilepath)  # Save the uploaded file

        # Extract voice embedding for the current audio
        new_voice_embedding = extract_voice_embedding(trainfilepath, classifier)

        # Check if the voice embedding is already in the database
        if is_user_already_registered(new_voice_embedding, cursor):
            return jsonify({'msg': 'This user is already registered'}), 200

        # Register the user if not already in the database
        register_user(name, new_voice_embedding, cursor, conn)

        return jsonify({'msg': f"The user '{name}' was registered successfully."})

    return jsonify({'msg': 'Error in registration'}), 400

@app.route('/recognize', methods=['POST'])
def handle_recognition():
    audio_file = request.files['audio']

    # Normalize the path to ensure correct slashes
    filepath = os.path.normpath(os.path.join('C:/Users/hp/Downloads/Speaker_App', secure_filename(audio_file.filename)))
    print("filepath:", filepath)

    audio_file.save(filepath)

    # Check if the file exists and inspect its format
    if not os.path.exists(filepath):
        return jsonify({'msg': 'File not found after saving.'}), 400

    print("File saved at:", filepath)

    try:
        print("Loading the pre-trained model...")
        classifier = load_model()  # Load the pre-trained model
        testfilepath = filepath.replace("\\", "/")
        voice_embedding = extract_voice_embedding(testfilepath, classifier)
    except Exception as e:
        return jsonify({'msg': f"Failed to process audio file: {str(e)}"}), 500

    USERNAME = recognize_user(voice_embedding, cursor)

    return jsonify({'msg': 'Hey ' + USERNAME + ', Welcome to Stella Voice Assistant !!'})

if __name__ == '__main__':
    conn, cursor = init_db()
    classifier = load_model()  # Load the pre-trained model
    app.run(debug=True, port=5000)







