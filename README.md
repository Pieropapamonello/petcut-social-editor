# PetCut Studio

PetCut trasforma un protagonista, scene facoltative e una canzone in un montaggio verticale 9:16, esportato in MP4 a 576×1024 e 30 fps per Reels e TikTok.

Il motore analizza l'intero brano, sceglie l'estratto con il drop nella stessa posizione narrativa del riferimento e costruisce una griglia di beat, mezzi beat, quarti e onset. Tutti i tagli e gli accenti derivano da quella griglia. Anche con una sola foto o clip, il soggetto resta riconoscibile e gli sfondi cambiano tramite livelli, parallax e movimenti 2.5D.

## Montaggio unico

PetCut produce un solo stile, ricostruito sul riferimento fornito:

1. clip introduttive in movimento con nomi grandi;
2. roulette di pose scontornate su nero;
3. collage con la frase `CAN YOU IMAGINE FLOATING WEIGHTLESS`;
4. climax con protagonista persistente, sfondi separati, parallax, prospettiva, whip e flash sugli attacchi reali.

La durata consigliata è 24 secondi, come il riferimento. Se la struttura musicale non permette un montaggio completo, PetCut accorcia l'estratto o degrada esplicitamente la cue sheet senza creare segmenti fuori ordine. Le maschere con sfondo, fori, bordi rettangolari o identità incoerente vengono escluse; un matte non affidabile interrompe il job invece di ripetere mobili o pavimento in tutto il video.

Il primo upload è sempre il protagonista. Gli upload successivi sono soltanto scene o sfondi e non possono sostituirlo. Segmentazione, selezione dell'identità, rifinitura GrabCut, decontaminazione dei bordi e generazione dei livelli avvengono localmente; i file non vengono inviati a servizi AI esterni. Le informazioni sui modelli sono in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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
python -m compileall -q .
node --check static/app.js
```

## Deploy su Render

`render.yaml` è il Blueprint del progetto. In Render scegli **New → Blueprint**, collega il repository e conferma il servizio. Il Dockerfile installa FFmpeg, i font e tutte le dipendenze Python.

L’endpoint `GET /api/health` permette a Render di verificare lo stato del servizio. I file sorgente e i microclip sono temporanei; al termine del job rimangono soltanto l’MP4 da scaricare e pochi byte di metadati per recuperarne lo stato.

Variabili disponibili:

- `MAX_UPLOAD_MB` — limite totale della richiesta, predefinito a 200 MB.
- `OUTPUT_DIR` — directory dei render, predefinita a `data/exports`.
- `PETCUT_MODEL_DIR` — directory del modello di segmentazione, predefinita a `data/models`.
