import assemblyai as aai
import sounddevice as sd
import soundfile as sf
import requests
import os
import tempfile
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import simpleaudio as sa
import json

# Load API keys from .env file
load_dotenv()
ASSEMBLY_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Set AssemblyAI key
aai.settings.api_key = ASSEMBLY_API_KEY


def record_audio(filename, duration=5, fs=16000):
    print("🎙️ Recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)
    print("✅ Recording finished.")


def transcribe_audio(filepath):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(filepath)
    print(f"📝 Transcription: {transcript.text}")
    return transcript.text


def get_deepseek_response(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful AI created by DeepSeek. Answer briefly."},
        {"role": "user", "content": prompt},
    ]

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": messages,
        "temperature": 0.7,
        "stream": False
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"]
    print(f"🤖 DeepSeek Response: {reply}")
    return reply


def speak_text(response):
    client = ElevenLabs(api_key="ELEVEN_API_KEY")
    # Updated format for latest SDK
    audio_stream = client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    output_format="pcm_16000",  # Required format for direct playback
    text= response,
    model_id="eleven_multilingual_v2",
)

    audio_bytes = b"".join(audio_stream)

    # Play the audio directly using simpleaudio
    play_obj = sa.play_buffer(audio_bytes, num_channels=1, bytes_per_sample=2, sample_rate=16000)
    play_obj.wait_done()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        record_audio(temp_file.name, duration=5)
        transcript = transcribe_audio(temp_file.name)
        if transcript.strip():
            response = get_deepseek_response(transcript)
            speak_text(response)
        else:
            print("⚠️ No voice input detected.")
