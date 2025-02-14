import os
import uuid
import asyncio
import subprocess
import json
from zipfile import ZipFile
import stat
import gradio as gr
import ffmpeg
import cv2
import edge_tts
from googletrans import Translator
from huggingface_hub import HfApi
import moviepy.editor as mp
import spaces

# Constants and initialization
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "artificialguybr/video-dubbing"
MAX_VIDEO_DURATION = 60  # seconds

api = HfApi(token=HF_TOKEN)

# Extract and set permissions for ffmpeg
ZipFile("ffmpeg.zip").extractall()
st = os.stat('ffmpeg')
os.chmod('ffmpeg', st.st_mode | stat.S_IEXEC)

language_mapping = {
    'English': ('en', 'en-US-EricNeural'),
    'Spanish': ('es', 'es-ES-AlvaroNeural'),
    'French': ('fr', 'fr-FR-HenriNeural'),
    'German': ('de', 'de-DE-ConradNeural'),
    'Italian': ('it', 'it-IT-DiegoNeural'),
    'Portuguese': ('pt', 'pt-PT-DuarteNeural'),
    'Polish': ('pl', 'pl-PL-MarekNeural'),
    'Turkish': ('tr', 'tr-TR-AhmetNeural'),
    'Russian': ('ru', 'ru-RU-DmitryNeural'),
    'Dutch': ('nl', 'nl-NL-MaartenNeural'),
    'Czech': ('cs', 'cs-CZ-AntoninNeural'),
    'Arabic': ('ar', 'ar-SA-HamedNeural'),
    'Chinese (Simplified)': ('zh-CN', 'zh-CN-YunxiNeural'),
    'Japanese': ('ja', 'ja-JP-KeitaNeural'),
    'Korean': ('ko', 'ko-KR-InJoonNeural'),
    'Hindi': ('hi', 'hi-IN-MadhurNeural'),
    'Swedish': ('sv', 'sv-SE-MattiasNeural'),
    'Danish': ('da', 'da-DK-JeppeNeural'),
    'Finnish': ('fi', 'fi-FI-HarriNeural'),
    'Greek': ('el', 'el-GR-NestorasNeural')
}

print("Starting the program...")

def generate_unique_filename(extension):
    return f"{uuid.uuid4()}{extension}"

def cleanup_files(*files):
    for file in files:
        if file and os.path.exists(file):
            os.remove(file)
            print(f"Removed file: {file}")

@spaces.GPU(duration=90)
def transcribe_audio(file_path):
    print(f"Starting transcription of file: {file_path}")
    temp_audio = None
    
    if file_path.endswith(('.mp4', '.avi', '.mov', '.flv')):
        print("Video file detected. Extracting audio...")
        try:
            video = mp.VideoFileClip(file_path)
            temp_audio = generate_unique_filename(".wav")
            video.audio.write_audiofile(temp_audio)
            file_path = temp_audio
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            raise

    output_file = generate_unique_filename(".json")
    command = [
        "insanely-fast-whisper",
        "--file-name", file_path,
        "--device-id", "0",
        "--model-name", "openai/whisper-large-v3",
        "--task", "transcribe",
        "--timestamp", "chunk",
        "--transcript-path", output_file
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Transcription output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running insanely-fast-whisper: {e}")
        raise

    try:
        with open(output_file, "r") as f:
            transcription = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        raise

    result = transcription.get("text", " ".join([chunk["text"] for chunk in transcription.get("chunks", [])]))
    
    cleanup_files(output_file, temp_audio)
    
    return result

async def text_to_speech(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

@spaces.GPU
def process_video(video, target_language, use_wav2lip):
    try:
        if target_language is None:
            raise ValueError("Please select a Target Language for Dubbing.")
        
        run_uuid = uuid.uuid4().hex[:6]
        output_filename = f"{run_uuid}_resized_video.mp4"
        ffmpeg.input(video).output(output_filename, vf='scale=-2:720').run()

        video_path = output_filename
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Error: {video_path} does not exist.")

        video_info = ffmpeg.probe(video_path)
        video_duration = float(video_info['streams'][0]['duration'])

        if video_duration > MAX_VIDEO_DURATION:
            cleanup_files(video_path)
            raise ValueError(f"Video duration exceeds {MAX_VIDEO_DURATION} seconds. Please upload a shorter video.")

        ffmpeg.input(video_path).output(f"{run_uuid}_output_audio.wav", acodec='pcm_s24le', ar=48000, map='a').run()

        subprocess.run(f"ffmpeg -y -i {run_uuid}_output_audio.wav -af lowpass=3000,highpass=100 {run_uuid}_output_audio_final.wav", shell=True, check=True)
        
        whisper_text = transcribe_audio(f"{run_uuid}_output_audio_final.wav")
        print(f"Transcription successful: {whisper_text}")
                
        target_language_code, voice = language_mapping[target_language]
        translator = Translator()
        translated_text = translator.translate(whisper_text, dest=target_language_code).text
        print(f"Translated text: {translated_text}")

        asyncio.run(text_to_speech(translated_text, voice, f"{run_uuid}_output_synth.wav"))
        
        if use_wav2lip:
            try:
                subprocess.run(f"python Wav2Lip/inference.py --checkpoint_path 'Wav2Lip/checkpoints/wav2lip_gan.pth' --face '{video_path}' --audio '{run_uuid}_output_synth.wav' --pads 0 15 0 0 --resize_factor 1 --nosmooth --outfile '{run_uuid}_output_video.mp4'", shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Wav2Lip error: {str(e)}")
                gr.Warning("Wav2lip encountered an error. Falling back to simple audio replacement.")
                subprocess.run(f"ffmpeg -i {video_path} -i {run_uuid}_output_synth.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {run_uuid}_output_video.mp4", shell=True, check=True)
        else:
            subprocess.run(f"ffmpeg -i {video_path} -i {run_uuid}_output_synth.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {run_uuid}_output_video.mp4", shell=True, check=True)

        output_video_path = f"{run_uuid}_output_video.mp4"
        if not os.path.exists(output_video_path):
            raise FileNotFoundError(f"Error: {output_video_path} was not generated.")

        cleanup_files(
            f"{run_uuid}_resized_video.mp4",
            f"{run_uuid}_output_audio.wav",
            f"{run_uuid}_output_audio_final.wav",
            f"{run_uuid}_output_synth.wav"
        )

        return output_video_path, ""

    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        return None, f"Error: {str(e)}"

# Gradio interface setup
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Video Dubbing")
    gr.Markdown("This tool uses AI to dub videos into different languages. Upload a video, choose a target language, and get a dubbed version!")
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Upload Video")
            target_language = gr.Dropdown(
                choices=list(language_mapping.keys()), 
                label="Target Language for Dubbing", 
                value="Spanish"
            )
            use_wav2lip = gr.Checkbox(
                label="Use Wav2Lip for lip sync", 
                value=False, 
                info="Enable this if the video has close-up faces. May not work for all videos."
            )
            submit_button = gr.Button("Process Video", variant="primary")
        
        with gr.Column(scale=2):
            output_video = gr.Video(label="Processed Video")
            error_message = gr.Textbox(label="Status/Error Message")

    submit_button.click(
        process_video, 
        inputs=[video_input, target_language, use_wav2lip], 
        outputs=[output_video, error_message]
    )

    gr.Markdown("""
    ## Notes:
    - Video limit is 1 minute. The tool will dub all speakers using a single voice.
    - Processing may take up to 5 minutes.
    - This is an alpha version using open-source models.
    - Quality vs. speed trade-off was made for scalability and hardware limitations.
    - For videos longer than 1 minute, please duplicate this Space and adjust the limit in the code.
    """)

    gr.Markdown("""
    ---
        Hi
    """)

print("Launching Gradio interface...")
demo.queue()
demo.launch()