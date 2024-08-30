from flask import Flask, request, jsonify
import whisper
import os
import tempfile

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Check if an audio file is present in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    # Retrieve the audio file from the request
    audio_file = request.files['audio']
    
    # Save the audio file to a temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
    except Exception as e:
        return jsonify({"error": f"Failed to save audio file: {str(e)}"}), 500

    # Load and process the audio file
    try:
        # Load audio data from the temporary file
        audio = whisper.load_audio(temp_file_path)
        # Pad or trim the audio to fit the model input size
        audio = whisper.pad_or_trim(audio)
        # Convert audio to log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
    except Exception as e:
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # Detect the spoken language
    try:
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
    except Exception as e:
        return jsonify({"error": f"Failed to detect language: {str(e)}"}), 500

    # Decode the audio to text
    try:
        # Decode options can be customized; here using default options
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
    except Exception as e:
        return jsonify({"error": f"Failed to transcribe audio: {str(e)}"}), 500

    # Return the detected language and transcription as JSON
    return jsonify({
        "detected_language": detected_language,
        "transcription": result.text
    })

if __name__ == '__main__':
    app.run(debug=True)
