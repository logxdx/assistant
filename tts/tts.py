from openai import OpenAI
import pyaudio
import threading
import io
import time

class tts():
    def __init__(
            self,
            voice: str = "af_sarah",
            speed: float = 1.0,
            response_format: str = "pcm",
            base_url: str = "http://localhost:8880/v1",
            api_key: str = "not-needed",
        ):

        self.voice = voice
        self.api_key = api_key
        self.base_url = base_url
        self.response_format = response_format
        self.speed = speed
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.player = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
        )
        
        # Control flags for playback
        self.is_playing = False
        self.is_paused = False
        self.stop_requested = False
        self.audio_buffer = io.BytesIO()
        self.playback_thread = None
        self.current_position = 0
    
    def start(self, text: str):
        """Start tts playback of the given text"""
        # Stop any existing playback
        self.stop()
        
        # Reset state
        self.is_playing = True
        self.is_paused = False
        self.stop_requested = False
        self.audio_buffer = io.BytesIO()
        self.current_position = 0
        
        # Start playback in a new thread
        self.playback_thread = threading.Thread(target=self._playback_thread, args=(text,))
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def _playback_thread(self, text: str):
        """Worker thread to handle audio streaming and buffering"""
        # First, get the complete audio stream and store it
        with self.client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice=self.voice,
            input=text,
            response_format=self.response_format,
            speed=self.speed,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                if self.stop_requested:
                    return
                self.audio_buffer.write(chunk)
        
        # Reset buffer position for playback
        self.audio_buffer.seek(0)
        audio_data = self.audio_buffer.getvalue()
        buffer_size = 1024
        
        # Play the audio
        while self.current_position < len(audio_data) and not self.stop_requested:
            if not self.is_paused:
                end_pos = min(self.current_position + buffer_size, len(audio_data))
                chunk = audio_data[self.current_position:end_pos]
                self.player.write(chunk)
                self.current_position = end_pos
            else:
                # When paused, sleep briefly to avoid CPU hogging
                time.sleep(0.1)
        
        # Mark as not playing when done
        if not self.stop_requested:
            self.is_playing = False
    
    def stop(self):
        """Stop tts playback completely"""
        if self.is_playing:
            self.stop_requested = True
            self.is_playing = False
            self.is_paused = False
            
            # Wait for playback thread to finish
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=1.0)
            
            # Reset state
            self.current_position = 0
    
    def pause(self):
        """Pause tts playback"""
        if self.is_playing and not self.is_paused:
            self.is_paused = True
    
    def play(self):
        """Resume paused tts playback"""
        if self.is_playing and self.is_paused:
            self.is_paused = False

    def speak(self, text: str):
        """Speak the given text"""
        self.start(text)

    def shutdown(self):
        """Shutdown the tts system"""
        self.stop()
        self.player.stop_stream()
        self.player.close()
        pyaudio.PyAudio().terminate()

if __name__ == "__main__":
    speaker = tts(
        voice="am_echo",
        speed=1.2,
    )
    speaker.speak("Hello, this is a test of the tts system.")

    time.sleep(5)  # Wait for a while to let the speech finish
    speaker.stop()
