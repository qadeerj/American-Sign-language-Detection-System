import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load ML model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Text-to-Speech setup
engine = pyttsx3.init()

# Label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 
    28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: ' ', 37: '.'
}
expected_features = 42

# Initialize buffers and history
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""
last_registered_time = time.time()
registration_delay = 2.5  # Seconds between character registrations

# GUI Setup
root = tk.Tk()
root.title("American Sign Language Translator")
root.geometry("1400x800")
root.configure(bg="white")
root.resizable(False, False)

# Custom Colors and Fonts
PRIMARY_COLOR = "#2ecc71"    # Green
SECONDARY_COLOR = "#3498db"  # Blue
ACCENT_COLOR = "#e74c3c"     # Red
BG_COLOR = "white"           
CARD_COLOR = "#f8f9fa"       
TEXT_COLOR = "#2c3e50"       
BORDER_COLOR = "#dee2e6"     

TITLE_FONT = ("Helvetica", 32, "bold")
HEADING_FONT = ("Arial", 24, "bold")
BODY_FONT = ("Arial", 18)
BUTTON_FONT = ("Arial", 16, "bold")

# Header Section
header_frame = Frame(root, bg=BG_COLOR)
header_frame.pack(pady=20)

# Load logo (add your logo.png file)
try:
    logo_img = Image.open("logo.png").resize((80, 80))
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = Label(header_frame, image=logo_photo, bg=BG_COLOR)
    logo_label.grid(row=0, column=0, padx=10)
except:
    pass  # Continue without logo if file not found

title_label = Label(header_frame, 
                    text="American Sign Language Translator", 
                    font=TITLE_FONT, 
                    fg=TEXT_COLOR, 
                    bg=BG_COLOR)
title_label.grid(row=0, column=1, padx=10)

# Main Content Layout
main_frame = Frame(root, bg=BG_COLOR)
main_frame.pack(pady=20)

# Video Feed Section
video_container = Frame(main_frame, 
                        bg=BG_COLOR, 
                        bd=4, 
                        relief="ridge",
                        width=640, 
                        height=480,
                        highlightbackground=BORDER_COLOR,
                        highlightthickness=2)
video_container.grid(row=0, column=0, padx=40, pady=10)
video_container.grid_propagate(False)

video_label = Label(video_container, bg=BG_COLOR)
video_label.pack(expand=True, fill="both")

# Results Panel
results_frame = Frame(main_frame, bg=BG_COLOR)
results_frame.grid(row=0, column=1, padx=40, pady=10)

# Detection Card
detection_card = Frame(results_frame, 
                      bg=CARD_COLOR, 
                      width=400,
                      height=200,
                      highlightbackground=BORDER_COLOR,
                      highlightthickness=1)
detection_card.pack(pady=20, fill="x")
Label(detection_card, 
      text="CURRENT DETECTION", 
      font=HEADING_FONT, 
      bg=CARD_COLOR, 
      fg=TEXT_COLOR).pack(pady=10)
current_alphabet = StringVar(value="---")
Label(detection_card, 
      textvariable=current_alphabet, 
      font=("Arial", 48, "bold"), 
      bg=CARD_COLOR, 
      fg=PRIMARY_COLOR).pack(pady=10)

# Word Card
word_card = Frame(results_frame, 
                 bg=CARD_COLOR, 
                 width=400,
                 height=150,
                 highlightbackground=BORDER_COLOR,
                 highlightthickness=1)
word_card.pack(pady=20, fill="x")
Label(word_card, 
      text="CURRENT WORD", 
      font=HEADING_FONT, 
      bg=CARD_COLOR, 
      fg=TEXT_COLOR).pack(pady=5)
current_word = StringVar(value="---")
Label(word_card, 
      textvariable=current_word, 
      font=BODY_FONT, 
      bg=CARD_COLOR, 
      fg=SECONDARY_COLOR).pack(pady=5)

# Sentence Card
sentence_card = Frame(results_frame, 
                     bg=CARD_COLOR, 
                     width=400,
                     height=200,
                     highlightbackground=BORDER_COLOR,
                     highlightthickness=1)
sentence_card.pack(pady=20, fill="x")
Label(sentence_card, 
      text="TRANSLATED SENTENCE", 
      font=HEADING_FONT, 
      bg=CARD_COLOR, 
      fg=TEXT_COLOR).pack(pady=5)
current_sentence = StringVar(value="---")
Label(sentence_card, 
      textvariable=current_sentence, 
      font=BODY_FONT, 
      bg=CARD_COLOR, 
      fg=TEXT_COLOR, 
      wraplength=380, 
      justify="left").pack(pady=5)

# Control Buttons
button_frame = Frame(root, bg=BG_COLOR)
button_frame.pack(pady=30)

def reset_sentence():
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    current_word.set("---")
    current_sentence.set("---")
    current_alphabet.set("---")

def toggle_pause():
    if pause_button.cget('text') == "Pause":
        pause_button.config(text="Resume")
    else:
        pause_button.config(text="Pause")

def speak_text(text):
    def tts_thread():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=tts_thread, daemon=True).start()

# Buttons
Button(button_frame, 
       text="Reset Session", 
       font=BUTTON_FONT, 
       command=reset_sentence,
       bg=ACCENT_COLOR, 
       fg="white", 
       padx=20,
       pady=10,
       bd=0,
       activebackground="#c0392b").grid(row=0, column=0, padx=15)

pause_button = Button(button_frame, 
                     text="Pause", 
                     font=BUTTON_FONT, 
                     command=toggle_pause,
                     bg=SECONDARY_COLOR, 
                     fg="white", 
                     padx=20,
                     pady=10,
                     bd=0,
                     activebackground="#2980b9")
pause_button.grid(row=0, column=1, padx=15)

Button(button_frame, 
       text="Speak Sentence", 
       font=BUTTON_FONT, 
       command=lambda: speak_text(current_sentence.get()),
       bg=PRIMARY_COLOR, 
       fg="white", 
       padx=20,
       pady=10,
       bd=0,
       activebackground="#27ae60").grid(row=0, column=2, padx=15)

# Hover Effects
def on_enter(e):
    e.widget['bg'] = e.widget.cget('activebackground')
def on_leave(e):
    original_color = ACCENT_COLOR if e.widget.cget('text') == "Reset Session" else \
                   SECONDARY_COLOR if e.widget.cget('text') == "Pause" else PRIMARY_COLOR
    e.widget['bg'] = original_color

for btn in button_frame.winfo_children():
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

# Video Processing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    if pause_button.cget('text') == "Resume":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, process_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            if len(data_aux) < expected_features:
                data_aux.extend([0]*(expected_features-len(data_aux)))
            elif len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            stabilization_buffer.append(predicted_character)
            if len(stabilization_buffer) > 30:
                stabilization_buffer.pop(0)

            if stabilization_buffer.count(predicted_character) > 25:
                current_time = time.time()
                if current_time - last_registered_time > registration_delay:
                    stable_char = predicted_character
                    last_registered_time = current_time
                    current_alphabet.set(stable_char)

                    if stable_char == ' ':
                        if word_buffer.strip():
                            speak_text(word_buffer)
                            sentence += word_buffer + " "
                            current_sentence.set(sentence.strip())
                        word_buffer = ""
                        current_word.set("---")
                    elif stable_char == '.':
                        if word_buffer.strip():
                            speak_text(word_buffer)
                            sentence += word_buffer + "."
                            current_sentence.set(sentence.strip())
                        word_buffer = ""
                        current_word.set("---")
                    else:
                        word_buffer += stable_char
                        current_word.set(word_buffer)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    cv2.putText(frame, f"Current Sign: {current_alphabet.get()}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 120, 200), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)
    root.after(10, process_frame)

# Start Application
process_frame()
root.mainloop()