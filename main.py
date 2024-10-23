from flask import Flask, request, jsonify, session, render_template
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json
import whisper
from speaker_reg_app import (
    recognize_user, record_audio, register_user, 
    load_model, init_db, extract_voice_embedding, 
    is_user_already_registered
)
from config import AUDIO_DIR
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/register')
def register_page():
    return render_template('register.html')  

@app.route('/transcription')
def transcription_page():
    return render_template('transcription.html') 
@app.route('/recognize')
def recognize_page():
    return render_template('recognize.html') 


@app.route('/register', methods=['POST'])
def handle_register():
    name = request.form.get('name')
    audio_file = request.files.get('audio')
    print("Name:", name)
    print("Audio File:", audio_file)

    if name and audio_file:
        trainfilepath = os.path.join(AUDIO_DIR, 'train.wav')
        print("Train File Path:", trainfilepath)
        audio_file.save(trainfilepath)
        new_voice_embedding = extract_voice_embedding(trainfilepath, classifier)
        if is_user_already_registered(new_voice_embedding, cursor):
            return jsonify({'msg': 'This user is already registered'}), 200
        result = model.transcribe(trainfilepath)
        transcription = result['text']
        print("Transcription:", transcription)
        register_user(name, new_voice_embedding, cursor, conn)
        return jsonify({'msg': f"The user '{name}' was registered successfully.", 'transcription': transcription})
    return jsonify({'msg': 'Error in registration'}), 400

@app.route('/recognize', methods=['POST'])
def handle_recognition():
    audio_file = request.files['audio']
    testfilepath = os.path.join(AUDIO_DIR, 'test.wav')
    filepath = testfilepath
    print("filepath:", filepath)
    audio_file.save(filepath)
    if not os.path.exists(filepath):
        return jsonify({'msg': 'File not found after saving.'}), 400
    print("File saved at:", filepath)
    result = model.transcribe(filepath)
    transcription = result['text']
    print("Transcription:", transcription)
    try:
        print("Loading the pre-trained model...")
        classifier = load_model()  # Load the pre-trained model
        testfilepath = filepath.replace("\\", "/")
        voice_embedding = extract_voice_embedding(testfilepath, classifier)
    except Exception as e:
        return jsonify({'msg': f"Failed to process audio file: {str(e)}"}), 500
    USERNAME = recognize_user(voice_embedding, cursor)
    return jsonify({'msg': 'Hey ' + USERNAME + ', Welcome to Stella Voice Assistant !!', 'transcription': transcription})
model = whisper.load_model("base")  # You can choose 'tiny', 'base', 'small', 'medium', or 'large' model depending on your requirement

@app.route('/transcription', methods=['POST'])
def handle_transcription():
    try:
        audio_file = request.files['audio']
        filepath = os.path.normpath(os.path.join('C:/Users/hp/Downloads/Speaker_App', secure_filename(audio_file.filename)))
        print("filepath:", filepath)
        audio_file.save(filepath)
        if not os.path.exists(filepath):
            return jsonify({'msg': 'File not found after saving.'}), 400
        print("File saved at:", filepath)
        print("Transcribing audio using Whisper model...")
        result = model.transcribe(filepath)
        transcription = result['text']
        print("Transcription:", transcription)
        return jsonify({'msg': 'Transcription successful', 'transcription': transcription})
    except Exception as e:
        return jsonify({'msg': f"Failed to transcribe audio file: {str(e)}"}), 500

if __name__ == '__main__':
    conn, cursor = init_db()
    classifier = load_model()  
    app.run(debug=True, port=5000)







