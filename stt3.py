import streamlit as st
from openai import OpenAI
import os
from io import BytesIO

# Set up the OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Whisper Transcription Testing App")

# Sidebar for advanced settings
st.sidebar.header("Advanced Settings")

# Model Selection (only "whisper-1" available)
model = "whisper-1"
st.sidebar.write("**Model**: whisper-1")

# Temperature
temperature = st.sidebar.slider(
    "Temperature (controls randomness)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

# Format
output_format = st.sidebar.selectbox("Output Format", ["text", "json"])

# Timestamps (for json output only)
timestamp_granularity = st.sidebar.selectbox(
    "Timestamp Granularity (only for JSON format)",
    ["none", "segment", "word", "both"],
    index=0
) if output_format == "json" else "none"


# Function to limit the prompt to 224 tokens (approx. 1,500 characters)
def limit_prompt(prompt, max_tokens=224):
    return prompt[:1500] if len(prompt) > 1500 else prompt

# Initialize session state for storing the prompt
if "saved_prompt" not in st.session_state:
    st.session_state.saved_prompt = ""

# Prompt input in the main section
prompt_text = st.text_area("Enter a prompt to guide the transcription (optional, max 224 tokens):")
submit_prompt = st.button("Submit Prompt")

# Save the prompt when "Submit Prompt" is clicked
if submit_prompt:
    st.session_state.saved_prompt = limit_prompt(prompt_text)
    st.success("Prompt saved successfully!")

# Option to record or upload audio
option = st.selectbox("Choose an option:", ("Record Audio", "Upload Audio"))

audio_file = None

if option == "Record Audio":
    # Attempt to use the experimental audio input
    try:
        audio_file = st.experimental_audio_input("Record your audio")
    except AttributeError:
        st.warning("Your Streamlit version doesn't support audio recording. Please upload an audio file instead.")
elif option == "Upload Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

# Display audio file if available
if audio_file:
    st.audio(audio_file, format="audio/wav")

# Transcribe button to start transcription
transcribe_button = st.button("Transcribe")

# Transcribe audio when "Transcribe" button is clicked and an audio file is provided
if transcribe_button and audio_file:
    # Read the uploaded file and prepare it for transcription
    audio_bytes = audio_file.read()
    audio_data = BytesIO(audio_bytes)
    audio_data.name = audio_file.name  # Set the name attribute for MIME type recognition

    # Define granularity for JSON output
    granularity_options = {
        "none": None,
        "segment": ["segment"],
        "word": ["word"],
        "both": ["segment", "word"]
    }
    timestamp_options = granularity_options[timestamp_granularity]

    # Transcribe audio using the client object
    try:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_data,
            prompt=st.session_state.saved_prompt or "",  # Use saved prompt if available, else empty
            response_format="verbose_json" if output_format == "json" else "text",
            temperature=temperature,
            language="en",  # Default to English, or adapt this if dynamic language selection is added
            timestamp_granularities=timestamp_options if timestamp_options else None
        )
        
        # Display the transcription result based on the output format
        if output_format == "text":
            st.write("Transcription:")
            st.write(transcription)
        elif output_format == "json":
            # Handle 'verbose_json' output by accessing attributes directly
            st.write("Transcription JSON:")
            st.json(transcription)

            # Display word-level timestamps if available
            if timestamp_granularity in ["word", "both"] and hasattr(transcription, "words"):
                st.write("Word-Level Timestamps:")
                for word_info in transcription.words:
                    st.write(word_info)
        else:
            st.error("Unexpected response format from the API.")
    except Exception as e:
        st.error(f"Error in transcription: {e}")
elif transcribe_button and not audio_file:
    st.warning("Please upload or record an audio file to transcribe.")
