# PetCut Social Editor

Web app Flask che trasforma una o più foto/video e una canzone in un MP4 verticale 9:16 (576×1024), pronto per Reels e TikTok.

L'app analizza sia il BPM musicale sia gli attacchi del brano. Se la canzone è lenta, usa un tempo di montaggio più rapido e aggancia tagli, flash e cambi d'inquadratura agli onset rilevati. L'interfaccia suggerisce quanti contenuti distinti usare, ma funziona anche con una sola foto o un solo video.

Preset disponibili:

- **Cinematic Zoom**
- **Fast Beat Edit**
- **CapCut Collage**
- **Floating Cutout**, ispirato agli edit social con soggetti scontornati su nero

Floating Cutout usa una timeline in quattro atti: mini-intro, roulette di sagome, collage di parole e climax a tutto schermo. Un rilevatore leggero identifica persone e animali; GrabCut e l'analisi temporale ripuliscono lo sfondo. Il primo render Floating può richiedere qualche secondo in più perché scarica il modello compatto del rilevatore.

## Avvio locale

Serve Docker, oppure Python 3.12 e FFmpeg installati.

```bash
docker build -t petcut .
docker run -p 10000:10000 petcut
```

Apri `http://localhost:10000`.

## Deploy su Render

Il file `render.yaml` è un Blueprint. Su Render scegli **New > Blueprint**, collega questo repository e conferma il servizio Docker. Render userà FFmpeg incluso nel Dockerfile.

I file caricati, i fotogrammi intermedi e i microclip sono temporanei. Al termine del montaggio PetCut conserva soltanto l'MP4 finale del job.
