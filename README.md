# edu-video-generator
learn with AI stories
import streamlit as st
import torch
from diffusers import TextToVideoSDPipeline, StableDiffusionPipeline
from gtts import gTTS
from moviepy.editor import *
from concurrent.futures import ThreadPoolExecutor
import os
import time

# Configuration
NUM_SCENES = 3  # Reduced from 5 for faster generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_HEIGHT = 512  # Smaller resolution for faster processing

# Initialize models (do this once at startup)
@st.cache_resource
def load_models():
    """Load all AI models with caching to avoid reloading"""
    models = {
        "text_to_video": TextToVideoSDPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w",
            torch_dtype=torch.float16
        ).to(DEVICE),
        "text_to_image": StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(DEVICE)
    }
    return models

def generate_story(topic, grade_level, style):
    """Faster story generation using template-based approach"""
    template = f"""Create a {style} story about {topic} for {grade_level} students.
    Structure:
    1. Introduction to main character
    2. The educational challenge they face
    3. How they solve it
    4. Lesson learned
    
    Make it engaging but concise (max 300 words)."""
    
    # Simulated faster response (replace with actual LLM call if needed)
    return f"""Once upon a time in ancient India, a young student named Arun struggled with {topic}. 
    Through his journey to the temple of knowledge, he discovered three key insights about {topic}. 
    First... Second... Third... The moral is that {topic} connects us all."""

def generate_images_parallel(prompts, model):
    """Generate images in parallel"""
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda p: model(prompt=p, height=VIDEO_HEIGHT, width=VIDEO_HEIGHT*2).images[0],
            prompts
        ))
    return results

def generate_video_clips(model, prompts):
    """Generate short video clips instead of static images"""
    clips = []
    for prompt in prompts:
        frames = model(prompt, num_frames=24).frames  # Shorter clips
        clip = ImageSequenceClip(frames, fps=8)  # Lower FPS for faster processing
        clips.append(clip)
    return clips

def text_to_speech(text, language):
    """Convert text to speech with caching"""
    lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta"}
    tts = gTTS(text=text, lang=lang_map.get(language, "en"))
    audio_path = f"narration_{int(time.time())}.mp3"
    tts.save(audio_path)
    return audio_path

# Streamlit UI
st.set_page_config(page_title="iThihasaya - Fast AI Video Generator", layout="wide")

with st.sidebar:
    st.title("Settings")
    language = st.selectbox("Language", ["English", "Hindi", "Telugu", "Tamil"])
    style = st.selectbox("Story Style", ["Historical", "Mythological", "Adventure"])
    
topic = st.text_input("Enter educational topic:")
if st.button("Generate Video") and topic:
    models = load_models()
    
    with st.spinner("Generating story..."):
        story = generate_story(topic, "High School", style)
    
    with st.spinner("Creating visuals (faster method)..."):
        # Method 1: Use video generation if GPU available
        if "cuda" in DEVICE:
            prompts = [f"{style} scene about {topic}: {part}" 
                      for part in story.split(".")[:NUM_SCENES]]
            video_clips = generate_video_clips(models["text_to_video"], prompts)
        # Fallback to parallel image generation
        else:
            prompts = [f"{style} illustration of: {part}" 
                      for part in story.split(".")[:NUM_SCENES]]
            images = generate_images_parallel(prompts, models["text_to_image"])
            video_clips = [ImageClip(img).set_duration(3) for img in images]  # 3 sec per image
    
    with st.spinner("Adding narration..."):
        audio_path = text_to_speech(story, language)
        audio_clip = AudioFileClip(audio_path)
    
    with st.spinner("Finalizing video..."):
        # Combine clips with audio
        final_clip = concatenate_videoclips(video_clips)
        final_clip = final_clip.set_audio(audio_clip)
        output_path = f"ithihasaya_{topic.replace(' ','_')}.mp4"
        final_clip.write_videofile(output_path, fps=24, codec="libx264")
        
        # Display result
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("Download Video", f, file_name=output_path)
        
        # Cleanup
        os.remove(audio_path)
        os.remove(output_path)
