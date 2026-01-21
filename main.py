# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
import json
import numpy as np
from typing import Dict, List
from pydub import AudioSegment
import torchaudio
import torch
from speechbrain.pretrained import SpeakerRecognition
from dotenv import load_dotenv
import google.generativeai as genai
import datetime as dt

# ======================================================
# ENV
# ======================================================
load_dotenv()
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

if not ASSEMBLYAI_KEY or not GEMINI_KEY or not SECRET_KEY:
    raise RuntimeError("Clés API manquantes")

# ======================================================
# FASTAPI
# ======================================================
app = FastAPI(title="API Analyse Réunion IA", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# SPEAKER MODEL
# ======================================================
model_spk = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec"
)

# ======================================================
# UTILS
# ======================================================
def convertion_en_wav(input_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    output = f"/tmp/{uuid.uuid4().hex}.wav"
    audio.export(output, format="wav")
    return output


def extract_embedding(audio_segment):
    tmp = f"/tmp/{uuid.uuid4().hex}.wav"
    audio_segment.export(tmp, format="wav")
    waveform, _ = torchaudio.load(tmp)
    os.remove(tmp)
    waveform = waveform[:1]
    with torch.no_grad():
        emb = model_spk.encode_batch(waveform)
    return emb.squeeze(0)


def cosine_similarity(e1, e2):
    e1 = e1.squeeze().cpu().numpy()
    e2 = e2.squeeze().cpu().numpy()
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))


# ======================================================
# TRANSCRIPTION
# ======================================================
def transcribe_meeting(audio_path):
    import assemblyai as aai
    aai.settings.api_key = ASSEMBLYAI_KEY
    config = aai.TranscriptionConfig(speaker_labels=True, language_code="fr")
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)

    segments = []
    for u in transcript.utterances:
        segments.append({
            "start": round(u.start / 1000, 2),
            "end": round(u.end / 1000, 2),
            "text": u.text,
            "speaker": u.speaker
        })
    return segments


# ======================================================
# SPEAKER MATCHING
# ======================================================
def map_speakers(segments, audio_path, reference_embeddings, threshold=0.80):
    audio = AudioSegment.from_file(audio_path)
    mapped = []

    for seg in segments:
        part = audio[int(seg["start"]*1000):int(seg["end"]*1000)]
        emb = extract_embedding(part)

        best_name = "Inconnu"
        best_score = 0

        for name, ref_emb in reference_embeddings.items():
            score = cosine_similarity(emb, ref_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < threshold:
            best_name = "Inconnu"

        seg["real_speaker"] = best_name
        seg["similarity_score"] = best_score
        mapped.append(seg)

    return mapped


# ======================================================
# PARTICIPATION
# ======================================================
def compute_participation(mapped_segments, participants):
    stats = {p: {"time": 0, "turns": 0} for p in participants}

    for s in mapped_segments:
        p = s["real_speaker"]
        if p in stats:
            stats[p]["time"] += s["end"] - s["start"]
            stats[p]["turns"] += 1

    for p, v in stats.items():
        if v["turns"] >= 5:
            v["niveau"] = "Élevé"
            v["score"] = 9
        elif v["turns"] >= 2:
            v["niveau"] = "Modéré"
            v["score"] = 6
        elif v["turns"] == 1:
            v["niveau"] = "Faible"
            v["score"] = 3
        else:
            v["niveau"] = "Nul"
            v["score"] = 0

    return stats


def segments_by_speaker(mapped_segments, participants):
    grouped = {p: "" for p in participants}
    grouped["Inconnu"] = ""

    for s in mapped_segments:
        grouped[s["real_speaker"]] += s["text"] + " "

    return grouped


# ======================================================
# GEMINI ANALYSIS
# ======================================================
def analyze_with_llm(mapped_segments, participants_roles):
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    grouped = segments_by_speaker(mapped_segments, list(participants_roles.keys()))

    prompt = f"""
Tu es un assistant pédagogique bienveillant spécialisé dans l'accompagnement
des étudiants et des enseignants lors de réunions académiques.

Ton objectif n'est pas de juger, mais d'aider chaque participant à progresser,
en valorisant les points positifs et en formulant les axes d'amélioration de
manière douce, constructive et encourageante.


Participants et rôles :
{json.dumps(participants_roles, ensure_ascii=False)}

Transcriptions par participant :
{json.dumps(grouped, ensure_ascii=False)}

Retourne UNIQUEMENT un JSON valide avec cette structure exacte :

{{
  "analyse_participants": [
    {{
      "nom": "",
      "role": "",
      "evaluation_participation": {{
        "temps_parole": "",
        "tours_parole": 0
      }},
      "evaluation_role": {{
        "conformite": "",
        "note": ""
      }},
      "emotion_dominante": "",
      "constat_individuel": "",
      "recommandations_encouragements": ""
    }}
  ],
  "analyse_dynamique_collective": {{
    "constat": "",
    "recommandations_globales": []
  }}
}}

Même si un participant n'a pas parlé, remplis tous les champs.
Règles de style obligatoires :
- Utilise un ton calme, bienveillant et pédagogique
- Évite toute formulation brutale, sèche ou culpabilisante
- Privilégie les phrases nuancées : "peut être renforcé", "gagnerait à", "il serait intéressant de"
- Valorise toujours au moins un point positif, même en cas de faible participation
- Si un participant n'a pas parlé, ne le blâme pas : propose des encouragements

"""

    response = model.generate_content(prompt)
    txt = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(txt)


# ======================================================
# ENDPOINT MODIFIÉ POUR ACCEPTER voice_references
# ======================================================
@app.post("/analyze_meeting/")
async def analyze_meeting(
    audio: UploadFile = File(...),
    participants_roles: str = Form(...),
    voice_references: List[UploadFile] = File(default=[])
):
    participants_roles = {
        k.strip(): v for k, v in json.loads(participants_roles).items()
    }
    participants = list(participants_roles.keys())

    # Save main audio file
    tmp_audio = f"/tmp/{uuid.uuid4().hex}_{audio.filename}"
    with open(tmp_audio, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Process voice references from uploaded files
    reference_embeddings = {}
    
    for voice_file in voice_references:
        # Extract participant name from filename (format: Prenom_Nom.wav)
        filename = voice_file.filename
        participant_name = os.path.splitext(filename)[0].replace('_', ' ')
        
        print(f"Processing voice reference for: {participant_name}")
        
        # Save voice file temporarily
        tmp_voice = f"/tmp/{uuid.uuid4().hex}_{filename}"
        with open(tmp_voice, "wb") as f:
            shutil.copyfileobj(voice_file.file, f)
        
        try:
            # Convert to wav and extract embedding
            wav_path = convertion_en_wav(tmp_voice)
            audio_seg = AudioSegment.from_file(wav_path)
            reference_embeddings[participant_name] = extract_embedding(audio_seg)
            print(f"Successfully extracted embedding for: {participant_name}")
            
            # Cleanup
            os.remove(tmp_voice)
            os.remove(wav_path)
        except Exception as e:
            print(f"Error processing voice reference for {participant_name}: {e}")
            if os.path.exists(tmp_voice):
                os.remove(tmp_voice)

    # Fallback: check local labellized_files for any missing participants
    for p in participants:
        if p not in reference_embeddings:
            ref = find_reference_audio_local(p)
            if ref:
                try:
                    wav = convertion_en_wav(ref)
                    audio_seg = AudioSegment.from_file(wav)
                    reference_embeddings[p] = extract_embedding(audio_seg)
                    print(f"Loaded local reference for: {p}")
                except Exception as e:
                    print(f"Error loading local reference for {p}: {e}")

    print(f"Total reference embeddings: {len(reference_embeddings)} for {len(participants)} participants")

    # Process meeting audio
    audio_wav = convertion_en_wav(tmp_audio)
    segments = transcribe_meeting(audio_wav)
    mapped = map_speakers(segments, audio_wav, reference_embeddings)
    participation = compute_participation(mapped, participants)
    analysis = analyze_with_llm(mapped, participants_roles)
    grouped = segments_by_speaker(mapped, participants)

    # Cleanup
    os.remove(tmp_audio)
    os.remove(audio_wav)

    return JSONResponse({
        "status": "success",
        "report": {
            "roles": participants_roles,
            "participation": participation,
            "analyse_ia": {
                **analysis,
                "grouped_segments": grouped
            },
            "constat_global": "Analyse IA complète de la réunion",
            "date": str(dt.date.today())
        }
    })


def find_reference_audio_local(participant_name):
    """Fallback: search in local labellized_files directory"""
    if not os.path.exists("labellized_files"):
        return None
    tokens = participant_name.lower().split()
    for f in os.listdir("labellized_files"):
        fname = f.lower()
        if all(token in fname for token in tokens):
            return os.path.join("labellized_files", f)
    return None


@app.get("/ping")
async def ping():
    return {"status": "ok"}
