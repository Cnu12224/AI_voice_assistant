import assemblyai as aai
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import requests
import simpleaudio as sa
import os
from dotenv import load_dotenv
import json

class AIVoiceAgent:
    def __init__(self):
        load_dotenv()
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_url = "https://openrouter.ai/api/v1/chat/completions"  # Example URL

        self.transcriber = None

        self.full_transcript = [
            {"role": "system", "content": "You are a language model called R1 created by DeepSeek, answer the questions being asked in less than 300 characters."},
        ]

    def start_transcription(self):
        print(f"\nReal-time transcription: ", end="\r\n")
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
        )
        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
        self.transcriber.stream(microphone_stream)

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        return

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            print(transcript.text)
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        return

    def on_close(self):
        return

    def generate_ai_response(self, transcript):
        self.stop_transcription()

        self.full_transcript.append({"role": "user", "content": transcript.text})
        print(f"\nUser: {transcript.text}", end="\r\n")

        # ---- DeepSeek API CALL ----
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",  # or your exact model name
            "messages": self.full_transcript,
            "temperature": 0.7,
            "stream": False  # No streaming
        }

        response = requests.post(self.deepseek_url, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            print("Error from DeepSeek:", response.text)
            return

        response_json = response.json()
        response_text = response_json['choices'][0]['message']['content']

        # ---- ElevenLabs Text-to-Speech ----
        print("DeepSeek R1:", end="\r\n")

        text_buffer = ""
        full_text = ""
        for sentence in response_text.split(". "):
            sentence = sentence.strip()
            if not sentence.endswith("."):
                sentence += "."
            audio_stream = self.client.generate(
                text=sentence,
                model="eleven_turbo_v2",
                stream=True
            )
            print(sentence, end="\n", flush=True)
            self.play_audio_stream(audio_stream)
            full_text += sentence + " "


    def play_audio_stream(self, audio_stream):
        wave_obj = sa.WaveObject.from_wave_file(audio_stream)
        play_obj = wave_obj.play()
        play_obj.wait_done()


        self.full_transcript.append({"role": "assistant", "content": full_text.strip()})
        self.start_transcription()


ai_voice_agent = AIVoiceAgent()
ai_voice_agent.start_transcription()
