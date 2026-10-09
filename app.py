
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.io.wavfile import write
import io

# ------------------------------------------------
# LOAD AI MODEL
# ------------------------------------------------

model = joblib.load("tinnitus_ai_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")


# ------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------

st.set_page_config(
    page_title="AI Powered Tinnitus Relief System",
    page_icon="🎧",
    layout="centered"
)


# ------------------------------------------------
# PROFESSIONAL WEBSITE THEME
# ------------------------------------------------

st.markdown("""
<style>
/* Light background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F4F8FC !important;
}

/* Main title and section headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp [data-testid="stHeader"] {
    color: #123B5D !important;
}

/* Main title */
.stApp h1 {
    text-align: center !important;
    font-weight: 700 !important;
}

/* Body text */
.stApp p,
.stApp label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #334E68 !important;
}

/* Input fields */
.stApp input {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    -webkit-text-fill-color: #1F2937 !important;
}

/* Dropdown fields */
.stApp [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #B8C9D6 !important;
}

.stApp [data-baseweb="select"] span {
    color: #1F2937 !important;
}

/* Buttons */
.stApp .stButton > button {
    background-color: #176B87 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stApp .stButton > button p {
    color: #FFFFFF !important;
}

/* Alerts and dividers */
.stApp [data-testid="stAlert"] {
    border-radius: 8px !important;
}

.stApp hr {
    border-top: 1px solid #D5E2EC !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 AI Powered Personalized Tinnitus Relief System")

st.write(
    "Enter the patient parameters below to generate a "
    "prototype AI-based therapy recommendation."
)

st.divider()


# ------------------------------------------------
# PATIENT INPUT
# ------------------------------------------------

st.subheader("Patient Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=22
)

tinnitus_type = st.selectbox(
    "Tinnitus Type",
    ["Ringing", "Buzzing", "Hissing"]
)

tinnitus_side = st.selectbox(
    "Tinnitus Side",
    ["Left", "Right", "Both"]
)

# ------------------------------------------------
# TINNITUS FREQUENCY MATCHING TEST
# ------------------------------------------------

st.subheader("Tinnitus Frequency Matching Test")

st.write(
    "Listen to the sound samples below and select the frequency "
    "that most closely resembles the pitch of the tinnitus sound you hear."
)

from scipy.io.wavfile import write
from io import BytesIO
import numpy as np

# Available test frequencies
frequency_options = [1000, 2000, 3000, 4000, 5000, 6000, 8000]

sample_rate = 44100
test_duration = 2

# Create audio samples
for freq in frequency_options:

    t = np.linspace(
        0,
        test_duration,
        int(sample_rate * test_duration),
        endpoint=False
    )

    audio = 0.05 * np.sin(
        2 * np.pi * freq * t
    )

    audio_int16 = np.int16(
        audio * 32767
    )

    buffer = BytesIO()

    write(
        buffer,
        sample_rate,
        audio_int16
    )

    st.write(f"🎵 {freq} Hz")

    st.audio(
        buffer.getvalue(),
        format="audio/wav"
    )


# Patient selects the closest frequency
frequency_hz = st.selectbox(
    "Which frequency is closest to your tinnitus sound?",
    frequency_options,
    index=None,
    placeholder="Select your tinnitus frequency"
)

st.success(
    f"Selected Tinnitus Frequency: {frequency_hz} Hz"
)

duration = st.number_input(
    "Duration (Months)",
    min_value=1,
    max_value=120,
    value=8
)

# ------------------------------------------------
# THI QUESTIONNAIRE
# ------------------------------------------------

st.subheader("Tinnitus Handicap Assessment")

st.write(
    "Answer the following questions based on how tinnitus "
    "affects you."
)

thi_questions = [
    "Does tinnitus make it difficult for you to concentrate?",
    "Does tinnitus make you feel frustrated?",
    "Does tinnitus affect your ability to relax?",
    "Does tinnitus interfere with your daily activities?",
    "Does tinnitus make it difficult to sleep?"
]

thi_score = 0

for i, question in enumerate(thi_questions):

    answer = st.radio(
    question,
    ["Yes", "Sometimes", "No"],
    index=None,
    key=f"thi_{i}"
)

    if answer == "Yes":
        thi_score += 4

    elif answer == "Sometimes":
        thi_score += 2

    elif answer == "No":
        thi_score += 0


# Scale prototype questionnaire score to 0–100
thi_score = int((thi_score / 20) * 100)

st.info(f"Calculated THI Score: {thi_score}")
hearing_difficulty = st.selectbox(
    "Hearing Difficulty",
    ["None", "Mild", "Moderate", "Severe"]
)


# ------------------------------------------------
# GENERATE BUTTON
# ------------------------------------------------

# ------------------------------------------------

if st.button("🤖 GENERATE PERSONALIZED THERAPY"):

    # Create patient DataFrame
    patient_input = pd.DataFrame([{
        "Age": age,
        "Tinnitus_Type": tinnitus_type,
        "Tinnitus_Side": tinnitus_side,
        "Frequency_Hz": frequency_hz,
        "Duration_Months": duration,
        "THI_Score": thi_score,
        "Hearing_Difficulty": hearing_difficulty
    }])

    # Encode categorical data
    for column in [
        "Tinnitus_Type",
        "Tinnitus_Side",
        "Hearing_Difficulty"
    ]:
        patient_input[column] = (
            label_encoders[column]
            .transform(patient_input[column])
        )

    # AI prediction
    prediction = model.predict(patient_input)

    recommended_therapy = (
        label_encoders["Recommended_Therapy"]
        .inverse_transform(prediction)[0]
    )

    # ------------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------------

    st.success("Therapy recommendation generated!")

    st.subheader("🤖 AI Recommendation")

    st.write(
        "### Recommended Therapy:",
        recommended_therapy
    )

    st.write(
        "### Personalized Frequency:",
        frequency_hz,
        "Hz"
    )


    # ------------------------------------------------
    # AUDIO GENERATION
    # ------------------------------------------------

    sample_rate = 44100
    audio_duration = 10
    volume = 0.05

    samples = int(sample_rate * audio_duration)

    t = np.linspace(
        0,
        audio_duration,
        samples,
        endpoint=False
    )

    if recommended_therapy == "White_Noise":

        audio = np.random.normal(
            0,
            1,
            samples
        )

    elif recommended_therapy == "Pink_Noise":

        white = np.random.normal(
            0,
            1,
            samples
        )

        audio = np.cumsum(white)

    elif recommended_therapy == "Narrow_Band_Noise":

        noise = np.random.normal(
            0,
            1,
            samples
        )

        carrier = np.sin(
            2 * np.pi * frequency_hz * t
        )

        audio = noise * carrier

    elif recommended_therapy == "Nature_Sound":

        audio = (
            np.sin(2 * np.pi * 200 * t)
            + 0.5 * np.sin(2 * np.pi * 400 * t)
        )


    # Normalize audio
    audio = audio / np.max(np.abs(audio))

    # Apply low prototype volume
    audio = audio * volume

    # Convert to 16-bit
    audio_int16 = np.int16(
        audio * 32767
    )


    # Save audio in memory
    audio_buffer = io.BytesIO()

    write(
        audio_buffer,
        sample_rate,
        audio_int16
    )

    audio_buffer.seek(0)


    # ------------------------------------------------
    # AUDIO OUTPUT
    # ------------------------------------------------

    st.subheader("🎧 Personalized Therapy Audio")

    st.audio(
        audio_buffer,
        format="audio/wav"
    )

    st.download_button(
        label="⬇ Download Personalized Audio",
        data=audio_buffer,
        file_name=(
            f"personalized_{recommended_therapy}_"
            f"{frequency_hz}Hz.wav"
        ),
        mime="audio/wav"
    )


# ------------------------------------------------
# DISCLAIMER
# ------------------------------------------------

st.divider()

st.caption(
    "⚠️ This is an educational prototype and does not "
    "diagnose, treat, or guarantee relief from tinnitus. "
    "Audio output is not clinically calibrated."
)
