from faster_whisper import WhisperModel

_whisper_model = None

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model

def transcribe_audio(file_path):
    model = _get_whisper_model()
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        initial_prompt=(
            "Bu ses kaydı otomotiv recall ve şikayet sorularıyla ilgilidir. "
            "Geçebilecek marka isimleri: Nissan, Audi, Volkswagen, BMW, Mercedes, "
            "Toyota, Ford, Honda, Hyundai, Kia, Chevrolet, Juke, Qashqai, X-Trail,Micra. Geçebilecek terimler: "
            "recall, geri çağırma, şikayet, complaint, VIN, model yılı."
        ),
    )
    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip(), info.language