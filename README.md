# PetCut Studio

PetCut trasforma foto o video e una canzone in un montaggio verticale 9:16, esportato in MP4 a 576×1024 e 30 fps per Reels e TikTok.

Il motore non applica un effetto casuale in modo uniforme: analizza BPM, transienti, intensità e punto di drop, poi costruisce una timeline diversa per ogni sezione del brano. I tagli sono quantizzati sui frame e agganciati agli onset più vicini. Anche un solo contenuto può essere riutilizzato con inquadrature, pose, crop e movimenti differenti.

## Preset

- **Animal Roulette** — quattro atti: etichette, roulette di cutout, frase cinetica e climax full-frame.
- **Mystery Reveal** — silhouette frammentata, `? / ?? / ???`, rivelazione al drop e montaggio finale.
- **Kinetic Strips** — titolo condensato dietro al soggetto, pannelli diagonali, card, maschere V/X e strisce.
- **Beat Montage** — montaggio più semplice con clip leggibili, punch/whip brevi e cambi ogni due beat.

Per i preset con livelli, il soggetto viene individuato localmente con un modello ONNX incluso nel repository. Non vengono inviati file a servizi AI esterni. Le informazioni sul modello sono in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Avvio locale

Il modo più semplice è Docker:

```bash
docker build -t petcut-studio .
docker run --rm -p 10000:10000 petcut-studio
```

Apri `http://localhost:10000`.

In alternativa servono Python 3.12 e FFmpeg:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Test

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py audio_analysis.py render_engine.py
node --check static/app.js
```

## Deploy su Render

`render.yaml` è il Blueprint del progetto. In Render scegli **New → Blueprint**, collega il repository e conferma il servizio. Il Dockerfile installa FFmpeg, i font e tutte le dipendenze Python.

L’endpoint `GET /api/health` permette a Render di verificare lo stato del servizio. I file sorgente e i microclip sono temporanei; al termine del job rimangono soltanto l’MP4 da scaricare e pochi byte di metadati per recuperarne lo stato.

Variabili disponibili:

- `MAX_UPLOAD_MB` — limite totale della richiesta, predefinito a 200 MB.
- `OUTPUT_DIR` — directory dei render, predefinita a `data/exports`.
- `PETCUT_MODEL_DIR` — directory del modello di segmentazione, predefinita a `data/models`.
