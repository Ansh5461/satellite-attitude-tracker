import soundfile as sf

path = "/media/ansh/SDA1/its-onnn/data/move2_351020_2018-12-05/raw/satnogs_351020_2018-12-05T14-45-34.ogg"  # fill in real filename

info = sf.info(path)
print("samplerate:", info.samplerate)
print("channels:", info.channels)
print("duration (s):", info.duration)
print("frames:", info.frames)
print("format:", info.format, info.subtype)

data, sr = sf.read(path, frames=5*info.samplerate)  # just read first 5s
print("data shape:", data.shape)
print("dtype:", data.dtype)
print("first few samples:\n", data[:5])