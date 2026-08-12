# PetCut Social Editor

Web app Flask che trasforma una o più foto/video e una canzone in un MP4 verticale 9:16 con zoom ritmico, color grading e testo opzionale. Include tre preset: Cinematic Zoom, Fast Beat Edit e CapCut Collage.

Quando si seleziona una canzone, l'app stima il BPM e suggerisce il numero ideale di foto/clip per la durata e il preset scelti. Anche un solo contenuto è sufficiente: PetCut lo ripete in scene con movimento e variazioni sincronizzate al ritmo.

## Avvio locale

Serve Docker, oppure Python 3.12 e FFmpeg installati.

```bash
docker build -t petcut .
docker run -p 10000:10000 petcut
```

Apri `http://localhost:10000`.

## Deploy su Render

Il file `render.yaml` è un Blueprint. Su Render scegli **New > Blueprint**, collega questo repository e conferma il servizio Docker. Render userà FFmpeg incluso nel Dockerfile.

Nota: i file caricati e generati sono temporanei. Per produzione e file grandi è consigliato collegare uno storage persistente (es. Cloudinary o S3/R2).
