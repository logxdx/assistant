import os
import string
import time

import markdown
from bs4 import BeautifulSoup
from colorama import Fore, Style

from brain import get_assistant_model, get_response, greet_me
from stt import stt
from tts import TTS, TTS_11Labs, tts

# Convert Markdown to plain text using BeautifulSoup
def markdown_to_plaintext(md_text):
    # Convert Markdown to HTML
    html_content = markdown.markdown(md_text)
    # Remove HTML tags to get plain text
    soup = BeautifulSoup(html_content, "html.parser")
    plaintext = soup.get_text()
    return plaintext

# play audio file on wakeword activation
def play_audio_file(file_path):
    os.system(
        f"ffplay -nodisp -autoexit -hide_banner -loglevel panic -i {file_path} -volume 200"
    )

if __name__ == "__main__":
    os.system("cls")

    NAME = "logX"

    # Get the assistant model
    assistant = get_assistant_model()
    assistant_name = assistant["name"]
    on_sound = "notifs/ripple.mp3"

    INPUT_MODE = "VOICE"
    OUTPUT_MODE = "VOICE"

    speaker = None
    if OUTPUT_MODE == "VOICE":
        # Initialize the speaker
        print(Fore.LIGHTYELLOW_EX + "Initializing TTS...", flush=True)
        speaker = tts(**assistant["tts_config"])

    recorder = None
    if INPUT_MODE == "VOICE":
        # Initialize the recorder
        print(Fore.LIGHTYELLOW_EX + "Initializing STT...", flush=True)
        recorder = stt(
            model="medium.en",
            language="en",
            wake_word=assistant["keyword"],
            openwakeword_model_paths=[assistant["oww_path"]],
            wakeword_backend="oww",
            silero_sensitivity=0.3,
            beam_size=3,
            post_speech_silence_duration=1.5,
            on_vad_detect_start=lambda: (
                speaker.pause(),
                play_audio_file(on_sound),
                # print(Fore.RED + "\n" + "VAD started 👂", flush=True),
            ),
            # on_vad_detect_stop=lambda: (
            #     print(Fore.RED + "\n" + "VAD stopped 🛑", flush=True)
            # ),
            # on_recording_start=lambda: (
            #     print(Fore.LIGHTBLUE_EX + "\n" + "Listening 🎤", flush=True)
            # ),
            # on_recording_stop=lambda: (
            #     print(Fore.LIGHTBLUE_EX + "\n" + "Recording complete ✅", flush=True)
            # ),
        )

    # Initialize the conversation
    conversation = []

    print(Fore.LIGHTCYAN_EX + assistant_name + ": ", flush=True)
    greetings = ""
    for text in greet_me(name=NAME, assistant_name=assistant_name):
        print(text, end="", flush=True)
        greetings += text
    print("\n")
    speaker.speak(greetings)

    conversation.append({"role": "assistant", "content": str(greetings)})

    try:
        while True:
            if INPUT_MODE == "VOICE" and recorder:
                command = ""
                try:
                    command = recorder.text()
                except Exception as e:
                    print(Fore.RED + f"Error: {e}")
                    continue
                print(Fore.LIGHTGREEN_EX + "User: ", end="", flush=True)
                print(f"{command}\n", flush=True)
            else:
                command = input("User: ")

            if command:
                if OUTPUT_MODE == "VOICE" and speaker:
                    speaker.stop()

                conversation.append({"role": "user", "content": str(command)})

                stop_command = command.translate(
                    str.maketrans("", "", string.punctuation)
                )
                if stop_command.lower() in [
                    "exit",
                    "quit",
                    "stop",
                    "bye",
                    "goodbye",
                    "turn off",
                    "shutdown",
                    "shut down",
                    "close",
                ]:
                    conversation = [
                        {
                            "role": "user",
                            "content": f"{assistant_name}, generate a short closing remark.",
                        }
                    ]
                    closing_response = ""
                    for text in get_response(conversation=conversation):
                        closing_response += text
                    print()

                    if OUTPUT_MODE == "VOICE" and speaker:
                        speaker.speak(closing_response)
                    if speaker:
                        speaker.shutdown()
                    if recorder:
                        recorder.shutdown()
                    Style.RESET_ALL
                    break

                print(Fore.CYAN + assistant_name + ": ", flush=True)
                full_response = ""
                stream = ""
                try:
                    stream = get_response(conversation=conversation)
                except Exception as e:
                    print(Fore.RED + f"Error: {e}")
                    continue

                for text in stream:
                    full_response += text
                full_response = markdown_to_plaintext(full_response)

                if "CLEAR" in full_response:
                    os.system("cls")
                    conversation.clear()

                if "SHUTDOWN" in full_response:
                    conversation = [
                        {
                            "role": "user",
                            "content": f"{assistant_name}, generate a short closing remark.",
                        }
                    ]
                    closing_response = ""
                    for text in get_response(conversation=conversation):
                        closing_response += text
                    print()

                    if OUTPUT_MODE == "VOICE" and speaker:
                        speaker.speak(closing_response)
                    if speaker:
                        speaker.shutdown()
                    if recorder:
                        recorder.shutdown()
                    Style.RESET_ALL
                    break

                if "STOPPED" not in full_response:
                    if "MODE:VOICE" in full_response:
                        OUTPUT_MODE = "VOICE"
                        if not speaker:
                            speaker = TTS(**assistant[2])
                        speaker.speak("Voice mode activated.")
                        continue
                    elif "MODE:TEXT" in full_response:
                        OUTPUT_MODE = "TEXT"
                        continue
                    if OUTPUT_MODE == "VOICE" and speaker:
                        speaker.speak(full_response)
                    if str(full_response).endswith("?"):
                        while speaker.is_playing:
                            time.sleep(0.5)
                        if INPUT_MODE == "VOICE" and recorder:
                            recorder.start()
            else:
                speaker.speak(f"Not {assistant_name} - Ignoring")

    except KeyboardInterrupt:
        os.system("cls")
        if speaker:
            speaker.shutdown()
        if recorder:
            recorder.shutdown()
        Style.RESET_ALL
        os.system("exit")
        for item in conversation:
            print()
            for key, value in item.items():
                print(f"{key}: {value}")
