#pip3 install soundfile sounddevice torchaudio
#pip3 install git+https://github.com/speechbrain/speechbrain.git@develop

from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
import sounddevice as sd
import soundfile as sf
import numpy as np
import torchaudio
import warnings
import sqlite3


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Database initialization
def init_db():
    conn = sqlite3.connect('C:/Users/hp/Downloads/Speaker_App/stella_db.sqlite', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS stellausers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            voice_embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    return conn, c

# Load embedding model
def load_model():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    return classifier

# Record audio from microphone using sounddevice
def record_audio(filename, duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to complete
    sf.write(filename, audio, sample_rate)  # Save the recorded audio to a file
    print(f"Recording finished: {filename}")
    return filename

# Extract speaker embedding
def extract_voice_embedding(filepath, classifier):
    signal, fs = torchaudio.load(filepath)
    embeddings = classifier.encode_batch(signal)
    return embeddings.numpy()[0][0]

# Register a new user
def register_user(name, voice_embedding, cursor, conn):
    cursor.execute("INSERT INTO stellausers (name, voice_embedding) VALUES (?, ?)", 
                   (name, voice_embedding.tobytes()))
    conn.commit()
    print(f"User {name} registered successfully.")

# Recognize an existing user
def recognize_user(voice_embedding, cursor):
    cursor.execute("SELECT id, name, voice_embedding FROM stellausers")
    users = cursor.fetchall()
    
    if not users:
        print("No users registered.")
        return None
    
    max_similarity = -1
    recognized_user = None
    
    for user in users:
        user_id, name, stored_embedding = user
        stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)

        # Calculate cosine similarity
        similarity = cosine_similarity([voice_embedding], [stored_embedding])[0][0]
        print(f"Similarity with {name}: {similarity:.2f}")

        if similarity > max_similarity:
            max_similarity = similarity
            recognized_user = name
    
    if max_similarity > 0.4:  # Similarity threshold for recognition
        print(f"***********************************")
        print(f"Hey {recognized_user}, How are you?")
        print(f"***********************************")
        return recognized_user
    else:
        print("No matching user found.")
        recognized_user = "New User"
        return recognized_user

# Main function
def main():
    conn, cursor = init_db()
    classifier = load_model()  # Load the pre-trained model
    while True:
        print("\n1. Register a new user")
        print("2. Recognize an existing user")
        print("3. Register and Test User")
        choice = input("Enter choice (1/2/3) or 'q' to quit: ")

        if choice == '1':
            name = input("Enter your name: ")
            print("Sample Words: MY VOICE IS MY PASSWORD")
            filepath = record_audio("C:/Users/hp/Downloads/Speaker_App/train.wav", duration=5)  # Record 5 seconds of audio
            voice_embedding = extract_voice_embedding(filepath, classifier)
            register_user(name, voice_embedding, cursor, conn)
        elif choice == '2':
            print("Sample Words: MY VOICE IS MY PASSWORD")
            filepath = record_audio("C:/Users/hp/Downloads/Speaker_App/test.wav", duration=5)  # Record 5 seconds of audio
            voice_embedding = extract_voice_embedding(filepath, classifier)
            recognize_user(voice_embedding, cursor)
        elif choice == '3':
            print("Sample Words: MY VOICE IS MY PASSWORD")
            name = input("Enter your name: ")
            filepath = record_audio("C:/Users/hp/Downloads/Speaker_App/train.wav", duration=5)  # Record 5 seconds of audio
            voice_embedding = extract_voice_embedding(filepath, classifier)
            register_user(name, voice_embedding, cursor, conn)

            filepath = record_audio("C:/Users/hp/Downloads/Speaker_App/test.wav", duration=5)  # Record 5 seconds of audio
            voice_embedding = extract_voice_embedding(filepath, classifier)
            print("Sample Words: MY VOICE IS MY PASSWORD")
            recognize_user(voice_embedding, cursor)
        elif choice.lower() == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 'q'.")
    
    conn.close()

# Check if user is already registered
def is_user_already_registered(new_embedding, cursor):
    cursor.execute("SELECT voice_embedding FROM stellausers")
    registered_users = cursor.fetchall()

    for user in registered_users:
        registered_embedding = np.frombuffer(user[0], dtype=np.float32)

        # Compare embeddings (cosine similarity)
        similarity = cosine_similarity([new_embedding], [registered_embedding])[0][0]
        
        if similarity > 0.9:  # Threshold for considering two embeddings as the same
            return True
    
    return False

if __name__ == "__main__":
    main()
