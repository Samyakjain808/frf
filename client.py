import requests

def send_audio_for_transcription(file_path):
    # Define the URL of the Flask server endpoint
    url = 'http://127.0.0.1:5000/transcribe'
    
    # Open the audio file in binary mode
    with open(file_path, 'rb') as audio_file:
        # Create a dictionary with the file to be sent
        files = {'audio': ('audio.mp3', audio_file)}
        # Send the POST request to the Flask server
        response = requests.post(url, files=files)
        
        # Return the JSON response from the server
        return response.json()

# Example usage
if __name__ == '__main__':
    result = send_audio_for_transcription('audio.mp3')
    print(result)
