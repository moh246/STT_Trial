import torchaudio
import torch
import time
from transformers import pipeline
# Load audio directly using torchaudio to bypass ffmpeg
waveform, sample_rate = torchaudio.load("sample_0.wav")
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
pipe_seamless = pipeline("automatic-speech-recognition", model="facebook/seamless-m4t-v2-large", trust_remote_code=True)
start_time = time.time()
transcription = pipe_seamless(waveform.squeeze(0).numpy(), generate_kwargs={"tgt_lang": "arb"})
end_time = time.time()
transcription_time = end_time - start_time
print(f"Transcription: {transcription['text']}")
print(f"Transcription Time: {transcription_time:.2f} seconds")