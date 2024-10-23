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

# trainfilepath = "C:/Users/hp/Downloads/Speaker_App/train.wav"
# testfilepath = "C:/Users/hp/Downloads/Speaker_App/test.wav"
# AUDIO_DIR = os.path.join(os.getcwd(), 'Audio_File')


@app.route('/')
def index():
    return render_template('index.html')  # Serve home page if needed.

@app.route('/register')
def register_page():
    return render_template('register.html')  # Serve the register page.


@app.route('/transcription')
def transcription_page():
    return render_template('transcription.html')  # Serve the transcription page.

@app.route('/recognize')
def recognize_page():
    return render_template('recognize.html')  # Serve the recognize page.

# @app.route('/register', methods=['POST'])
# def handle_register():
#     name = request.form.get('name')
#     audio_file = request.files.get('audio')
#     print("Name:", name)
#     print("Audio File:", audio_file)

#     if name and audio_file:
#         trainfilepath = "C:/Users/hp/Downloads/Speaker_App/train.wav"
#         audio_file.save(trainfilepath)  # Save the uploaded file

#         # Extract voice embedding for the current audio
#         new_voice_embedding = extract_voice_embedding(trainfilepath, classifier)

#         # Check if the voice embedding is already in the database
#         if is_user_already_registered(new_voice_embedding, cursor):
#             return jsonify({'msg': 'This user is already registered'}), 200
        
#         result = model.transcribe(trainfilepath)
#         transcription = result['text']
#         print("Transcription:", transcription)

#         # Register the user if not already in the database
#         register_user(name, new_voice_embedding, cursor, conn)

#         return jsonify({'msg': f"The user '{name}' was registered successfully.",'transcription': transcription})

#     return jsonify({'msg': 'Error in registration'}), 400



@app.route('/register', methods=['POST'])
def handle_register():
    name = request.form.get('name')
    audio_file = request.files.get('audio')
    print("Name:", name)
    print("Audio File:", audio_file)

    if name and audio_file:
        # Define paths for train.wav and test.wav inside the Audio_File folder
        trainfilepath = os.path.join(AUDIO_DIR, 'train.wav')
        print("Train File Path:", trainfilepath)
        
        
        # Save the uploaded file as train.wav (or you could change it based on input)
        audio_file.save(trainfilepath)

        # Extract voice embedding for the current audio
        new_voice_embedding = extract_voice_embedding(trainfilepath, classifier)

        # Check if the voice embedding is already in the database
        if is_user_already_registered(new_voice_embedding, cursor):
            return jsonify({'msg': 'This user is already registered'}), 200
        
        # Transcribe the saved audio file
        result = model.transcribe(trainfilepath)
        transcription = result['text']
        print("Transcription:", transcription)

        # Register the user if not already in the database
        register_user(name, new_voice_embedding, cursor, conn)

        return jsonify({'msg': f"The user '{name}' was registered successfully.", 'transcription': transcription})

    return jsonify({'msg': 'Error in registration'}), 400

@app.route('/recognize', methods=['POST'])
def handle_recognition():
    audio_file = request.files['audio']
    testfilepath = os.path.join(AUDIO_DIR, 'test.wav')

    # Normalize the path to ensure correct slashes
    filepath = testfilepath
    print("filepath:", filepath)

    audio_file.save(filepath)

    # Check if the file exists and inspect its format
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

        # Normalize the path to ensure correct slashes
        filepath = os.path.normpath(os.path.join('C:/Users/hp/Downloads/Speaker_App', secure_filename(audio_file.filename)))
        print("filepath:", filepath)

        # Save the uploaded audio file
        audio_file.save(filepath)

        # Check if the file exists and inspect its format
        if not os.path.exists(filepath):
            return jsonify({'msg': 'File not found after saving.'}), 400

        print("File saved at:", filepath)

        # Use the Whisper model to transcribe the audio
        print("Transcribing audio using Whisper model...")
        result = model.transcribe(filepath)
        transcription = result['text']
        print("Transcription:", transcription)

        # Return the transcription in the response
        return jsonify({'msg': 'Transcription successful', 'transcription': transcription})

    except Exception as e:
        return jsonify({'msg': f"Failed to transcribe audio file: {str(e)}"}), 500


if __name__ == '__main__':
    conn, cursor = init_db()
    classifier = load_model()  # Load the pre-trained model
    app.run(debug=True, port=5000)







