from paella import AcousticGenerator
from transformers import AutoModel, AutoConfig
import torch
import torchaudio
from einops import rearrange, repeat


#  CONFIG

num_quantizers = 4
sample_rate = 48000
num_channels = 2
num_frames = 480000
num_heads = 4
depth = 8
hidden_size = 768


# GET ENCODEC

config = AutoConfig.from_pretrained(
    "Audiogen/encodec",
    trust_remote_code=True,
    sample_rate=sample_rate,
    num_quantizers=num_quantizers * 2,
    num_channels=num_channels,
)
print(config)
encodec = AutoModel.from_config(config, trust_remote_code=True).cuda()


# INITIALIZE MODEL

model = AcousticGenerator(
    num_quantizers=num_quantizers,
    num_heads=num_heads,
    depth=depth,
    hidden_size=hidden_size,
).cuda()

# print num params
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params / 1e6:.2f}M")


# GET CODES

audio, sr = torchaudio.load('test.flac')
audio = torchaudio.functional.resample(audio, sr, sample_rate).mean(0, keepdim=True)
audio = audio[None, ..., :num_frames].cuda().repeat(1, num_channels, 1)
print(audio.shape)
codes = encodec.encode(audio)

codes = repeat(codes, 'b ... -> (b r) ...', r=8)
encodec_frames = codes.shape[-1]

assert codes.shape[1] == num_quantizers

print(f"codes shape: {codes.shape}")

# TRAIN LOOP

from tqdm import tqdm
from torch.optim import AdamW

optim = AdamW(model.parameters(), lr=1e-4)

steps = 500
losses = []
progress_bar = tqdm(range(steps))
for i in progress_bar:
    loss = model(
        codes,
    )
    optim.zero_grad()
    loss.backward()
    optim.step()
    losses.append(loss.item())
    progress_bar.set_description(f'loss: {loss.item():.4f}')
        

# plot losses
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()


# Sample
steps_scheduler = [16]
steps_scheduler.extend([2 for _ in range(num_quantizers - 1)])
gen_tokens = model.sample(
    seq_len=encodec_frames,
    steps_scheduler=steps_scheduler,
    mode='argmax',
)

#gen_tokens = rearrange(gen_tokens[:, 0, :], 'b (q s) -> b q s', q=num_quantizers)

print(f"Generated tokens shape: {gen_tokens.shape}")

# decode tokens
gen_audios = encodec.decode(gen_tokens)

print(f"Generated audios shape: {gen_audios.shape}")


# save audio

torchaudio.save("generated.flac", gen_audios[0].cpu(), sample_rate=sample_rate)