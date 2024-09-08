from moviepy.editor import VideoFileClip
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os, io
from flask_cors import CORS
from flask import Flask, request, render_template,jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'your_secret_string'
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp4','avi','mkv','flv','mov','mp3','wav','m4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
CORS(app)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#speech to text
# Helper function to extract audio from video
def extract_audio(video_path):
    audio_path = "temp_audio.wav"
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path, video_clip.duration

# Speech-to-text function
def speech_to_text(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Set up the pipeline correctly
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Ensure the audio file is correctly loaded and processed
    with open(audio_path, "rb") as audio_file:
        audio_input = audio_file.read()
    
    # Use the pipeline to transcribe the audio
    result = pipe(audio_input,return_timestamps=True)
    os.remove(audio_path)  # Clean up the temporary audio file
    
    return result['text']

def safe_file_delete(filepath):
    """Safely deletes a file, ensuring it exists first."""
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        print(f"File {filepath} not found, cannot delete.")
@app.route('/', methods=['GET'])
def index():
    return render_template('speechtotext.html')

@app.route('/speech-to-text', methods=['GET', 'POST'])
def speech_to_text_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "No selected file or file type not allowed"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        transcript = None  # Initialize transcript to None to ensure it has a value

        # Determine if the file is audio or video and process accordingly
        if filename.endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
            audio_path, _ = extract_audio(filepath)
            transcript = speech_to_text(audio_path)
            safe_file_delete(audio_path)  # Delete the extracted audio file
        elif filename.endswith(('.mp3', '.wav','.m4a')):
            transcript = speech_to_text(filepath)

        safe_file_delete(filepath)  # Delete the original uploaded file
        
        if transcript is None:
            return jsonify({"error": "Unsupported file type or empty transcript"}), 400

        return jsonify({"transcript": transcript})

    return render_template('speechtotext.html')

if __name__ == '__main__':
    app.run(debug=True)

