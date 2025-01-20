from fastapi import FastAPI, File, UploadFile, Form
from pydub import AudioSegment
import os
import whisper
import speech_recognition as sr

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), model: str = Form("whisper")):
    file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    with open(file_path, "wb") as f:
        f.write(await audio.read())

    try:
        audio_segment = AudioSegment.from_file(file_path)
        wav_path = file_path.replace(".mp3", ".wav")
        audio_segment.export(wav_path, format="wav")
    except Exception as e:
        return {"error": f"Error converting MP3 to WAV: {e}"}

    try:
        if model == "whisper":
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(file_path)
            transcription = result["text"]
        else:
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data)
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}
    finally:
        os.remove(file_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return {"transcription": transcription}
