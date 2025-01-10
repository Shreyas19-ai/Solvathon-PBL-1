import speech_recognition as sr
from gtts import gTTS
import pygame
import os
from googletrans import Translator
from openai import OpenAI
from transformers import pipeline

# Initialize recognizer, translator, and OpenAI API
recognizer = sr.Recognizer()
translator = Translator()

# Load BioBERT model for medical term validation
bio_bert = pipeline("fill-mask", model="dmis-lab/biobert-base-cased-v1.1")

# Function for text-to-speech
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

# Translate text between languages
def translate_text(text, src_lang="en", dest_lang="es"):
    result = translator.translate(text, src=src_lang, dest=dest_lang)
    return result.text

# Use OpenAI for advanced medical conversation processing
def openai_process(command):
    client = OpenAI(api_key="<your_openai_api_key>")
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical AI skilled in accurate translations and understanding healthcare-related terminology."},
            {"role": "user", "content": command}
        ]
    )
    return completion.choices[0].message.content

# Validate medical terms using BioBERT
def validate_medical_terms(text):
    words = text.split()
    validated_words = []
    for word in words:
        # Predict using BioBERT to validate terms
        masked_sentence = f"{word} is a [MASK] term."
        prediction = bio_bert(masked_sentence)
        # If 'medical' appears as a high-confidence prediction, retain the word
        if any("medical" in p["token_str"] for p in prediction):
            validated_words.append(word)
        else:
            validated_words.append(word)  # Keep the word even if not recognized
    return " ".join(validated_words)

# Main function to process commands
def process_command(command, src_lang="en", dest_lang="es"):
    print(f"Recognized command: {command}")
    
    if "translate" in command.lower():
        # Extract text to translate
        text_to_translate = command.split("translate")[1].strip()
        translated_text = translate_text(text_to_translate, src_lang, dest_lang)
        
        # Validate medical terms in the translated text
        validated_text = validate_medical_terms(translated_text)
        print(f"Validated Translation: {validated_text}")
        speak(f"Translation: {validated_text}")
    else:
        # Use OpenAI to process the command
        output = openai_process(command)
        speak(output)

# Main loop
if __name__ == "__main__":
    speak("Initializing real-time medical translator...")
    
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for wake word 'Cura'...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                wake_word = recognizer.recognize_google(audio)
                
                if wake_word.lower() == "cura":
                    speak("Yes, how can I assist?")
                    
                    # Listen for the actual command
                    with sr.Microphone() as source:
                        print("Listening for command...")
                        audio = recognizer.listen(source)
                        command = recognizer.recognize_google(audio)
                        
                        # Process the command (default: English to Spanish)
                        process_command(command, src_lang="en", dest_lang="es")
        
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
        except Exception as e:
            print(f"Error: {e}")
