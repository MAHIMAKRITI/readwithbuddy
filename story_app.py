# # import streamlit as st
# # from gtts import gTTS
# # from io import BytesIO
# # import base64
# # import os
# # import librosa
# # import numpy as np
# # import math

# # def text_to_speech(sentence):
# #     # Convert text to speech using gTTS
# #     tts = gTTS(text=sentence, lang='en')
# #     # Save speech as an audio file in memory
# #     fp = BytesIO()
# #     tts.write_to_fp(fp)
# #     return fp

# # def record_audio(filename, duration=6):
# #     import pyaudio
# #     import wave

# #     CHUNK = 1024
# #     FORMAT = pyaudio.paInt16
# #     CHANNELS = 1
# #     RATE = 44100
# #     RECORD_SECONDS = duration

# #     audio = pyaudio.PyAudio()

# #     stream = audio.open(format=FORMAT, channels=CHANNELS,
# #                         rate=RATE, input=True,
# #                         frames_per_buffer=CHUNK)

# #     st.write("Recording...")
# #     frames = []

# #     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
# #         data = stream.read(CHUNK)
# #         frames.append(data)

# #     st.write("Finished recording.")

# #     stream.stop_stream()
# #     stream.close()
# #     audio.terminate()

# #     with wave.open(filename, 'wb') as wf:
# #         wf.setnchannels(CHANNELS)
# #         wf.setsampwidth(audio.get_sample_size(FORMAT))
# #         wf.setframerate(RATE)
# #         wf.writeframes(b''.join(frames))

# #     return filename

# # def compute_similarity(audio_file1, audio_file2):
# #     # Load audio files
# #     y_ref, sr1 = librosa.load(audio_file1)
# #     y_comp, sr2 = librosa.load(audio_file2)

# #     # Compute chroma features
# #     chroma_ref = librosa.feature.chroma_cqt(y=y_ref, sr=sr1)
# #     chroma_comp = librosa.feature.chroma_cqt(y=y_comp, sr=sr2)

# #     # Use time-delay embedding for a cleaner similarity matrix
# #     x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
# #     x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=10, delay=3)

# #     # Compute the cross-similarity matrix
# #     similarity_matrix = librosa.segment.cross_similarity(x_comp, x_ref, metric='cosine')

# #     # Normalize the similarity matrix to get a score between 0 and 1
# #     similarity_score = similarity_matrix.mean()  # This is a simple way to get a single score

# #     return similarity_score

# # def main():
# #     st.title("Pronunciation Comparison")
    
# #     # Display original sentence
# #     original_sentence = "He has a big cat and dog and he lives down below the street in a big house."
# #     st.write("Original Sentence:", original_sentence)
    
# #     # Convert original sentence to audio
# #     original_audio_file = "original_audio.wav"
# #     original_audio_data = text_to_speech(original_sentence)
# #     with open(original_audio_file, 'wb') as f:
# #         f.write(original_audio_data.getvalue())
    
# #     # Button to listen to the original sentence
# #     if st.button("Listen to Original"):
# #         st.audio(original_audio_file, format='audio/wav')
    
# #     # Button to record user's pronunciation
# #     if st.button("Record Your Pronunciation"):
# #         # Record audio
# #         recorded_file = "recorded_audio.wav"
# #         record_audio(recorded_file)
# #         st.audio(recorded_file, format='audio/wav')
        
# #         # Compute similarity score
# #         similarity_score = compute_similarity(original_audio_file, recorded_file)
# #         st.write("Similarity Score:", similarity_score)
        
# #         # Determine number of filled stars
# #         num_filled_stars = math.ceil(similarity_score * 5)
        
# #         # Print filled stars
# #         stars = "★" * num_filled_stars
        
# #         # Print unfilled stars
# #         unfilled_stars = "☆" * (5 - num_filled_stars)
# #         #st.write(f"{stars}")
# #         #st.write(f"{unfilled_stars}")
# #         st.write(f"{stars}{unfilled_stars}")
        
# # if __name__ == "__main__":
# #     main()


# import streamlit as st
# from gtts import gTTS
# from io import BytesIO
# import base64
# import os
# import librosa
# import numpy as np
# import math

# def text_to_speech(sentence):
#     # Convert text to speech using gTTS
#     tts = gTTS(text=sentence, lang='en')
#     # Save speech as an audio file in memory
#     fp = BytesIO()
#     tts.write_to_fp(fp)
#     return fp

# def record_audio(filename, duration=6):
#     import pyaudio
#     import wave

#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 44100
#     RECORD_SECONDS = duration

#     audio = pyaudio.PyAudio()

#     stream = audio.open(format=FORMAT, channels=CHANNELS,
#                         rate=RATE, input=True,
#                         frames_per_buffer=CHUNK)

#     st.write("Recording...")
#     frames = []

#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.append(data)

#     st.write("Finished recording.")

#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(audio.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))

#     return filename

# def compute_similarity(audio_file1, audio_file2):
#     # Load audio files
#     y_ref, sr1 = librosa.load(audio_file1, sr=None)
#     y_comp, sr2 = librosa.load(audio_file2, sr=None)

#     # Compute MFCC features
#     mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr1)
#     mfcc_comp = librosa.feature.mfcc(y=y_comp, sr=sr2)

#     # Compute the cosine similarity between the mean of the MFCC features
#     similarity_score = np.dot(np.mean(mfcc_ref, axis=1), np.mean(mfcc_comp, axis=1)) / (
#         np.linalg.norm(np.mean(mfcc_ref, axis=1)) * np.linalg.norm(np.mean(mfcc_comp, axis=1))
#     )

#     # Ensure the similarity score is between 0 and 1
#     similarity_score = (similarity_score + 1) / 2

#     return similarity_score

# def main():
#     st.title("Pronunciation Comparison")
    
#     # Display original sentence
#     original_sentence = "He is a really good person but I hate him."
#     st.write("Original Sentence:", original_sentence)
    
#     # Convert original sentence to audio
#     original_audio_file = "original_audio.wav"
#     original_audio_data = text_to_speech(original_sentence)
#     with open(original_audio_file, 'wb') as f:
#         f.write(original_audio_data.getvalue())
    
#     # Button to listen to the original sentence
#     if st.button("Listen to Original"):
#         st.audio(original_audio_file, format='audio/wav')
    
#     # Button to record user's pronunciation
#     if st.button("Record Your Pronunciation"):
#         # Record audio
#         recorded_file = "recorded_audio.wav"
#         record_audio(recorded_file)
#         st.audio(recorded_file, format='audio/wav')
        
#         # Compute similarity score
#         similarity_score = compute_similarity(original_audio_file, recorded_file)
#         st.write("Similarity Score:", similarity_score)

#         # Determine number of filled stars
#         num_filled_stars = math.ceil(similarity_score * 5)
        
#         # Print filled stars
#         stars = "★" * num_filled_stars
        
#         # Print unfilled stars
#         unfilled_stars = "☆" * (5 - num_filled_stars)
#         st.write(f"{stars}{unfilled_stars}")

# if __name__ == "__main__":
#     main()

# import streamlit as st
# from gtts import gTTS
# from io import BytesIO
# import base64
# import os
# import librosa
# import numpy as np
# import speech_recognition as sr
# import math
# import Levenshtein

# def text_to_speech(sentence):
#     # Convert text to speech using gTTS
#     tts = gTTS(text=sentence, lang='en')
#     # Save speech as an audio file in memory
#     fp = BytesIO()
#     tts.write_to_fp(fp)
#     return fp

# def record_audio(filename, duration=6):
#     import pyaudio
#     import wave

#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 44100
#     RECORD_SECONDS = duration

#     audio = pyaudio.PyAudio()

#     stream = audio.open(format=FORMAT, channels=CHANNELS,
#                         rate=RATE, input=True,
#                         frames_per_buffer=CHUNK)

#     st.write("Recording...")
#     frames = []

#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.append(data)

#     st.write("Finished recording.")

#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(audio.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))

#     return filename

# def speech_to_text(audio_file):
#     # Initialize recognizer
#     recognizer = sr.Recognizer()

#     # Load audio file
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)

#     # Convert speech to text
#     try:
#         text = recognizer.recognize_google(audio_data)
#         return text
#     except sr.UnknownValueError:
#         st.write("Could not understand audio")
#         return ""
#     except sr.RequestError as e:
#         st.write(f"Could not request results; {e}")
#         return ""

# def compute_similarity(text1, text2):
#     # Compute Levenshtein distance
#     distance = Levenshtein.distance(text1.lower(), text2.lower())
    
#     # Normalize distance to get similarity score
#     similarity_score = 1 - (distance / max(len(text1), len(text2)))
    
#     return similarity_score

# def main():
#     st.title("Read with Buddy")
    
#     # Display original sentence
#     original_sentence = "He is a really good person but I hate him."
#     st.write("Original Sentence:", original_sentence)
    
#     # Convert original sentence to audio
#     original_audio_file = "original_audio.wav"
#     original_audio_data = text_to_speech(original_sentence)
#     with open(original_audio_file, 'wb') as f:
#         f.write(original_audio_data.getvalue())
    
#     # Button to listen to the original sentence
#     if st.button("Listen to Original"):
#         st.audio(original_audio_file, format='audio/wav')
    
#     # Button to record user's pronunciation
#     if st.button("Record Your Pronunciation"):
#         # Record audio
#         recorded_file = "recorded_audio.wav"
#         record_audio(recorded_file)
#         st.audio(recorded_file, format='audio/wav')
        
#         # Convert recorded audio to text
#         recorded_text = speech_to_text(recorded_file)
#         st.write("Recorded Text:", recorded_text)
        
#         # Compute similarity score
#         similarity_score = compute_similarity(original_sentence, recorded_text)
#         st.write("Similarity Score:", similarity_score)

#         # Determine number of filled stars
#         num_filled_stars = math.ceil(similarity_score * 5)
        
#         # Print filled stars
#         stars = "★" * num_filled_stars
        
#         # Print unfilled stars
#         unfilled_stars = "☆" * (5 - num_filled_stars)
#         st.write(f"{stars}{unfilled_stars}")

# if __name__ == "__main__":
#     main()

import streamlit as st
from gtts import gTTS
from io import BytesIO
import base64
import os
import librosa
import numpy as np
import speech_recognition as sr
import math
import Levenshtein
from PIL import Image


def text_to_speech(sentence):
    # Convert text to speech using gTTS
    tts = gTTS(text=sentence, lang='en')
    # Save speech as an audio file in memory
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp

def record_audio(filename, duration=6):
    import pyaudio
    import wave

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = duration

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    st.write("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Finished recording..")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return filename

def speech_to_text(audio_file):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    # Convert speech to text
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.write("Could not understand audio")
        return ""
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")
        return ""

def compute_similarity(text1, text2):
    # Compute Levenshtein distance
    distance = Levenshtein.distance(text1.lower(), text2.lower())
    
    # Normalize distance to get similarity score
    similarity_score = 1 - (distance / max(len(text1), len(text2)))
    
    return similarity_score

def main():

# Load the owl image
    owl_image = Image.open("owl_icon.png")

    # Resize the image to the desired height
    desired_height = 100
    aspect_ratio = owl_image.width / owl_image.height
    desired_width = int(desired_height * aspect_ratio)
    resized_image = owl_image.resize((desired_width, desired_height))

    
    # Create a layout with two columns
    col1, col2 = st.columns([4, 4])

    # Add the title "Read with Buddy" to the first column
    with col1:
        st.title("Read with Buddy")

    # Add the owl image to the second column
    with col2:
        st.image(resized_image)
    # st.title("Read with Buddy")
    # Add description
    st.write("Read with Buddy is an app that helps you improve your pronunciation by comparing your pronunciation of a sentence with the original one.")
    # st.write("                             ")
    st.write("----------------------------------------------------------------")
    # Display original sentence
    original_sentence1 = "He is a really good person but I hate him."
    st.write("Sentence 1:", original_sentence1)
    
    # Convert original sentence to audio
    original_audio_file1 = "original_audio1.wav"
    original_audio_data1 = text_to_speech(original_sentence1)
    with open(original_audio_file1, 'wb') as f:
        f.write(original_audio_data1.getvalue())
    
    # Button to listen to the original sentence
    if st.button("Listen 1"):
        st.audio(original_audio_file1, format='audio/wav')
    
    # Button to record user's pronunciation
    if st.button("Record Your Pronunciation 1"):
        # Record audio
        recorded_file1 = "recorded_audio1.wav"
        record_audio(recorded_file1)
        st.audio(recorded_file1, format='audio/wav')
        
        # Convert recorded audio to text
        recorded_text1 = speech_to_text(recorded_file1)
        st.write("Recorded Text 1:", recorded_text1)
        
        # Compute similarity score
        similarity_score1 = compute_similarity(original_sentence1, recorded_text1)
        # st.write("Similarity Score 1:", similarity_score1)

        # Determine number of filled stars
        num_filled_stars1 = math.ceil(similarity_score1 * 5)
        
        # Print filled stars
        stars1 = "★" * num_filled_stars1
        
        # Print unfilled stars
        unfilled_stars1 = "☆" * (5 - num_filled_stars1)
        st.write(f"{stars1}{unfilled_stars1}")
    st.write("----------------------------------------------------------------")
    # Display original sentence
    original_sentence2 = "She loves to sing and dance in the rain."
    st.write("Sentence 2:", original_sentence2)
    
    # Convert original sentence to audio
    original_audio_file2 = "original_audio2.wav"
    original_audio_data2 = text_to_speech(original_sentence2)
    with open(original_audio_file2, 'wb') as f:
        f.write(original_audio_data2.getvalue())
    
    # Button to listen to the original sentence
    if st.button("Listen 2"):
        st.audio(original_audio_file2, format='audio/wav')
    
    # Button to record user's pronunciation
    if st.button("Record Your Pronunciation 2"):
        # Record audio
        recorded_file2 = "recorded_audio2.wav"
        record_audio(recorded_file2)
        st.audio(recorded_file2, format='audio/wav')
        
        # Convert recorded audio to text
        recorded_text2 = speech_to_text(recorded_file2)
        st.write("Recorded Text 2:", recorded_text2)
        
        # Compute similarity score
        similarity_score2 = compute_similarity(original_sentence2, recorded_text2)
        # st.write("Similarity Score 2:", similarity_score2)

        # Determine number of filled stars
        num_filled_stars2 = math.ceil(similarity_score2 * 5)
        
        # Print filled stars
        stars2 = "★" * num_filled_stars2
        
        # Print unfilled stars
        unfilled_stars2 = "☆" * (5 - num_filled_stars2)
        st.write(f"{stars2}{unfilled_stars2}")
    st.write("----------------------------------------------------------------")
    # Display original sentence
    original_sentence3 = "I want to go to school."
    st.write("Sentence 3:", original_sentence3)
    
    # Convert original sentence to audio
    original_audio_file3 = "original_audio3.wav"
    original_audio_data3 = text_to_speech(original_sentence3)
    with open(original_audio_file3, 'wb') as f:
        f.write(original_audio_data3.getvalue())
    
    # Button to listen to the original sentence
    if st.button("Listen 3"):
        st.audio(original_audio_file3, format='audio/wav')
    
    # Button to record user's pronunciation
    if st.button("Record Your Pronunciation 3"):
        # Record audio
        recorded_file3 = "recorded_audio3.wav"
        record_audio(recorded_file3)
        st.audio(recorded_file3, format='audio/wav')
        
        # Convert recorded audio to text
        recorded_text3 = speech_to_text(recorded_file3)
        st.write("Recorded Text 3:", recorded_text3)
        
        # Compute similarity score
        similarity_score3 = compute_similarity(original_sentence3, recorded_text3)
        # st.write("Similarity Score 2:", similarity_score2)

        # Determine number of filled stars
        num_filled_stars3 = math.ceil(similarity_score3 * 5)
        
        # Print filled stars
        stars3 = "★" * num_filled_stars3
        
        # Print unfilled stars
        unfilled_stars3 = "☆" * (5 - num_filled_stars3)
        st.write(f"{stars3}{unfilled_stars3}")
if __name__ == "__main__":
    main()

