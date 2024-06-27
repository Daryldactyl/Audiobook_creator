import os
import torch
import torchaudio
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import streamlit as st
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from scipy.io.wavfile import write
import plotly.graph_objects as go

@st.cache_resource
def load_model():
    # st.write("Loading model...")
    config = XttsConfig()
    config.load_json("xtts/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir='xtts/')
    
    # st.write("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference_audio/IAmLegend_ep6 - 017.wav"])

    return model, gpt_cond_latent, speaker_embedding

@st.cache_resource
def download_punkt():
    # Download NLTK sentence tokenizer data
    nltk.download('punkt')

    
def split_sentences_with_punctuation(text):
    # Define regex pattern to match sentence-ending punctuation
    pattern = r'([.!?]+)'

    # Use regex to split text into sentences based on the pattern
    sentences = re.split(pattern, text)

    # Combine sentences with their respective punctuation marks
    combined_sentences = []
    i = 0
    while i < len(sentences) - 1:
        sentence = sentences[i].strip()
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
        combined_sentence = sentence + punctuation
        combined_sentences.append(combined_sentence)
        i += 2

    return combined_sentences

def create_audio(text, model, gpt_cond_latent, speaker_embedding, progress_bar):
    
    # Tokenize input text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize an empty list to collect sentence audio waveforms
    sentence_audio_waveforms = []

    # with st.sidebar:
    #     st.write("Synthesizing sentences:")
    #     progress_text = st.empty()
    
    # Iterate over each sentence and synthesize audio
    for i, sentence in enumerate(sentences):
        progress_text.markdown(f"Synthesizing sentence {i + 1}: {sentence}", unsafe_allow_html=True)
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(sentences))
    
        # Synthesize audio for the current sentence
        chunks = model.inference_stream(sentence, "en", gpt_cond_latent, speaker_embedding)
    
        # Collect all audio chunks for the current sentence
        sentence_audio_chunks = []
        for chunk in chunks:
            audio_numpy = chunk.squeeze().numpy()
            sentence_audio_chunks.append(audio_numpy)
    
        # Concatenate all audio chunks into a single waveform for the sentence
        sentence_audio_waveform = np.concatenate(sentence_audio_chunks)
    
        # Append the sentence audio waveform to the list of sentence waveforms
        sentence_audio_waveforms.append(sentence_audio_waveform)
    
    # Concatenate all sentence audio waveforms into a single continuous waveform
    audio_waveform = np.concatenate(sentence_audio_waveforms)
    
    # Save the audio waveform as a WAV file
    audio_file = "output.wav"
    write(audio_file, 24000, audio_waveform)

    return audio_file, audio_waveform

if __name__ == "__main__":
    col1,col3 = st.columns([3,2])
    with st.sidebar:
        st.write("Synthesizing sentences:")
        progress_text = st.empty()
    
    model, gpt_cond_latent, speaker_embedding = load_model()
    download_punkt()
    with col3:
        st.markdown('# Audio Goes here')
    with col1:
        text = st.text_area('Enter text for audiobook synthesis: ', height=500)
        progress_bar = st.progress(0)
        
        if st.button('Generate Audio'):
            audio_file, audio_waveform = create_audio(text, model, gpt_cond_latent, speaker_embedding, progress_bar)
            
            with col3:
                st.audio(audio_file)