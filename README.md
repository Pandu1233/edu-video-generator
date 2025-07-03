# edu-video-generator
learn with AI stories
import streamlit as st
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverSinglestepScheduler
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips, ImageClip
from concurrent.futures import ThreadPoolExecutor
import os
import time

# Configuration
NUM_SCENES = 3  # Reduced for faster generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_HEIGHT = 512  # Smaller resolution for faster processing
SAFETENSORS_FASTER_LOAD = True  # Use safetensors format for faster loading

# Initialize models with proper error handling
@st.cache_resource
def load_models():
    """Load all AI models with caching and proper configuration"""
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    
    try:
        models = {
            "text_to_video": DiffusionPipeline.from_pretrained(
                "cerspense/zeroscope_v2_576w",
                torch_dtype=torch_dtype,
                use_safetensors=SAFETENSORS_FASTER_LOAD,
                variant="fp16" if DEVICE == "cuda" else None
            ).to(DEVICE),
            
            "text_to_image": StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch_dtype,
                use_safetensors=SAFETENSORS_FASTER_LOAD
            ).to(DEVICE)
        }
        
        # Configure schedulers for stability
        models["text_to_video"].scheduler = DPMSolverSinglestepScheduler.from_config(
            models["text_to_video"].scheduler.config
        )
        
        return models
    
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None

def generate_story(topic, grade_level, style):
    """Generate story with proper formatting"""
    template = f"""Create a {style} story about {topic} for {grade_level} students.
    Structure:
    1. Introduction to main character
    2. The educational challenge they face
    3. How they solve it
    4. Lesson learned
    
    Make it engaging but concise (max 300 words)."""
    
    # Simulated response (replace with actual API call)
    return f"""Once upon a time in ancient India, a young student named Arun struggled with {topic}. 
    Through his journey to the temple of knowledge, he discovered three key insights about {topic}. 
    First... Second... Third... The moral is that {topic} connects us all."""

def generate_images_parallel(prompts, model):
    """Generate images in parallel with error handling"""
    def generate_single(prompt):
        try:
            return model(
                prompt=prompt,
                height=VIDEO_HEIGHT,
                width=VIDEO_HEIGHT*2,
                num_inference_steps=25  # Reduced for speed
            ).images[0]
        except Exception as e:
            st.warning(f"Failed to generate image for prompt: {prompt}\nError: {str(e)}")
            return None
    
    with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers for stability
        results = list(executor.map(generate_single, prompts))
    
    return [img for img in results if img is not None]

def generate_video_clips(model, prompts):
    """Generate short video clips with proper configuration"""
    clips = []
    for prompt in prompts:
        try:
            output = model(
                prompt,
                num_frames=24,  # Shorter clips
                num_inference_steps=25,  # Reduced steps
                height=VIDEO_HEIGHT,
                width=VIDEO_HEIGHT*2
            )
            frames = output.frames[0]  # Get frames from output
            clip = ImageSequenceClip(frames, fps=8)
            clips.append(clip)
        except Exception as e:
            st.warning(f"Failed to generate video for prompt: {prompt}\nError: {str(e)}")
    
    return clips

def text_to_speech(text, language):
    """Convert text to speech with caching and error handling"""
    lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta"}
    try:
        tts = gTTS(text=text[:500], lang=lang_map.get(language, "en"))  # Limit text length
        audio_path = f"narration_{int(time.time())}.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Failed to generate speech: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="iThihasaya - Stable AI Video Generator", layout="wide")

with st.sidebar:
    st.title("Settings")
    language = st.selectbox("Language", ["English", "Hindi", "Telugu", "Tamil"])
    style = st.selectbox("Story Style", ["Historical", "Mythological", "Adventure"])
    
topic = st.text_input("Enter educational topic:")
if st.button("Generate Video") and topic:
    models = load_models()
    if not models:
        st.error("Failed to initialize models. Please check your configuration.")
        st.stop()
    
    with st.spinner("Generating story..."):
        story = generate_story(topic, "High School", style)
        st.write(story)  # Show generated story
    
    with st.spinner("Creating visuals..."):
        prompts = [f"{style} scene about {topic}: {part}" 
                 for part in story.split(".")[:NUM_SCENES] if part.strip()]
        
        if DEVICE == "cuda":
            video_clips = generate_video_clips(models["text_to_video"], prompts)
        else:
            images = generate_images_parallel(prompts, models["text_to_image"])
            video_clips = [ImageClip(img).set_duration(3) for img in images]
        
        if not video_clips:
            st.error("Failed to generate any visuals")
            st.stop()
    
    with st.spinner("Adding narration..."):
        audio_path = text_to_speech(story, language)
        if not audio_path:
            st.error("Failed to generate audio")
            st.stop()
        
        try:
            audio_clip = AudioFileClip(audio_path)
        except Exception as e:
            st.error(f"Failed to load audio: {str(e)}")
            st.stop()
    
    with st.spinner("Finalizing video..."):
        try:
            final_clip = concatenate_videoclips(video_clips)
            final_clip = final_clip.set_audio(audio_clip)
            output_path = f"ithihasaya_{topic.replace(' ','_')}.mp4"
            final_clip.write_videofile(output_path, fps=24, codec="libx264")
            
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button("Download Video", f, file_name=output_path)
        except Exception as e:
            st.error(f"Failed to create final video: {str(e)}")
        finally:
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(output_path):
                os.remove(output_path)
