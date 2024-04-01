import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize


# Download the 'punkt' tokenizer model
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Record audio from the microphone
with sr.Microphone() as source:
    print("Please say something...")
    audio = recognizer.listen(source)

# Use Google Web Speech API to recognize speech
try:
    print("Recognizing speech...")
    text = recognizer.recognize_google(audio)
    print("You said:", text)

    # Tokenize the recognized text
    tokens = word_tokenize(text)

    # Perform NLP tasks (e.g., part-of-speech tagging, named entity recognition, etc.)
    # Example:
    tagged_tokens = nltk.pos_tag(tokens)
    print("Part-of-speech tagging:", tagged_tokens)

except sr.UnknownValueError:
    print("Sorry, could not understand audio.")
except sr.RequestError as e:
    print("Error fetching results; {0}".format(e))