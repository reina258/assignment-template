import torch
import numpy as np
from diffusers import DiffusionPipeline, AudioLDM2Pipeline, DPMSolverMultistepScheduler
import pyaudio

image_model = "runwayml/stable-diffusion-v1-5"
image_pipe = DiffusionPipeline.from_pretrained(image_model, torch_dtype=torch.float16)
image_pipe.to("cuda")

audio_pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float16)
audio_pipe.to("cuda")
audio_pipe.scheduler = DPMSolverMultistepScheduler.from_config(audio_pipe.scheduler.config)
audio_pipe.enable_model_cpu_offload()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, output=True)

while True:
    city = input("Which city would you like to travel to? (type 'exit' to quit): ")
    if city.lower() == 'exit':
        break

    landmark = input(f"What landmark or attraction would you like to visit in {city}?: ")

    activity = input(f"What would you like to do at {landmark} in {city}?: ")

    audio_prompt = f"Sounds and atmosphere of {landmark} in {city} while {activity}" 
    audios = audio_pipe(audio_prompt, num_inference_steps=200, audio_length_in_s=30).audios

    for audio in audios:
        stream.write(audio.astype(np.float32))

    image_prompt = f"A beautiful scene of {landmark} in {city} where someone is {activity}"
    images = image_pipe(image_prompt, num_inference_steps=20).images
    images[0].show() 

stream.stop_stream()
stream.close()
p.terminate()
