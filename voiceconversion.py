import os
import torch
import se_extractor
from api import ToneColorConverter

ckpt_converter = 'checkpoints/converter'
device = 'cuda:0'
prompt_dir = 'prompts'
skin_dir = 'skins'
output_dir = 'outputs'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(prompt_dir, exist_ok=True)
os.makedirs(skin_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

prompt = f"{prompt_dir}/prompt.wav"
prompt_se, audio_name = se_extractor.get_se(prompt, tone_color_converter, vad=True)

skin = f"{skin_dir}/skin.wav"
target_se, audio_name = se_extractor.get_se(skin, tone_color_converter, vad=True)

save_path = f'{output_dir}/output.wav'

# Run the tone color converter
tone_color_converter.convert(
    audio_src_path=prompt, 
    src_se=prompt_se, 
    tgt_se=target_se, 
    output_path=save_path)