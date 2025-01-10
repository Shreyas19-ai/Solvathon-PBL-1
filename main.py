import os
import webbrowser
import requests
from gtts import gTTS
import pygame
import speech_recognition as sr
from transformers import pipeline
from openai import OpenAI

# Initialize speech engine
def speak(text):
    tts = gTTS(text)
    tts.save("temp.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("temp.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()
    os.remove("temp.mp3")

# Initialize BioBERT pipeline
bio_bert = pipeline("fill-mask", model="dmis-lab/biobert-base-cased-v1.1")

def validate_medical_terms(text):
    """Validates medical terms in a given text using BioBERT."""
    words = text.split()
    validated_words = []

    for word in words:
        masked_sentence = f"{word} is a [MASK] term."
        predictions = bio_bert(masked_sentence)

        if any("medical" in p["token_str"] for p in predictions):
            validated_words.append(word)
        else:
            validated_words.append(word)  # Keep unverified terms for safety

    return " ".join(validated_words)

# Process commands with OpenAI
client = OpenAI(api_key="<your_openai_api_key>")

def ai_process(command):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "system", "content": "You are a virtual assistant named Jarvis skilled in general tasks like Alexa and Google Cloud. Give short responses please."},
            {"role": "user", "content": command}
        ]
    )
    return completion.choices[0].message.content

def process_command(c):
    if "open google" in c.lower():
        webbrowser.open("https://www.google.com")
    elif "open youtube" in c.lower():
        webbrowser.open("https://www.youtube.com")
    elif "open linkedin" in c.lower():
        webbrowser.open("https://www.linkedin.com")
    elif "news" in c.lower():
        api_key = "your_newsapi_key"
        r = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}")
        if r.status_code == 200:
            data = r.json()
            articles = data.get('articles', [])
            for article in articles:
                speak(article['title'])
    elif "validate" in c.lower():
        text_to_validate = c.split("validate", 1)[1].strip()
        validated_text = validate_medical_terms(text_to_validate)
        speak(f"Validated text: {validated_text}")
    else:
        output = ai_process(c)
        speak(output)

# Main program
if __name__ == "__main__":
    speak("Initializing Jarvis....")
    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for wake word 'Jarvis'...")
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=2)
                word = recognizer.recognize_google(audio)

                if word.lower() == "jarvis":
                    speak("Yes Sir")
                    print("Listening for command...")
                    audio = recognizer.listen(source)
                    command = recognizer.recognize_google(audio)
                    process_command(command)

        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Speech Recognition error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
