class_map = {"bosozoku": 0, "normal": 1}

chunk_duration = 3  # [s]
sr = 22050
n_mels = 128
n_fft = 2048
hop_length = 512

chunk_size = sr * chunk_duration
