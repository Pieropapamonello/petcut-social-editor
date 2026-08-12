const form = document.querySelector('#editor-form');
const mediaInput = document.querySelector('#media-input');
const audioInput = document.querySelector('#audio-input');
const mediaSummary = document.querySelector('#media-summary');
const audioSummary = document.querySelector('#audio-summary');
const durationSelect = document.querySelector('#duration-select');
const submitButton = document.querySelector('#submit-button');
const formError = document.querySelector('#form-error');

const analysisPanel = document.querySelector('#song-analysis');
const analysisEmpty = analysisPanel.querySelector('.analysis-empty');
const analysisResult = analysisPanel.querySelector('.analysis-result');
const analysisNote = document.querySelector('#analysis-note');
const musicBpm = document.querySelector('#music-bpm');
const editBpm = document.querySelector('#edit-bpm');
const contentCount = document.querySelector('#content-count');
const visualCuts = document.querySelector('#visual-cuts');
const presetPhaseList = document.querySelector('#preset-phase-list');

const renderPanel = document.querySelector('#render-panel');
const renderHeading = document.querySelector('#render-heading');
const renderProgress = document.querySelector('#render-progress');
const renderStatus = document.querySelector('#render-status');
const progressPercent = document.querySelector('#progress-percent');
const renderSteps = [...document.querySelectorAll('[data-render-step]')];

const resultPanel = document.querySelector('#result-panel');
const resultHeading = document.querySelector('#result-heading');
const resultSummary = document.querySelector('#result-summary');
const resultVideo = document.querySelector('#result-video');
const downloadButton = document.querySelector('#download-button');
const newEditButton = document.querySelector('#new-edit-button');

const presetPhases = {
  animal_roulette: ['Etichette', 'Roulette cutout', 'Frase cinetica', 'Climax'],
  mystery_reveal: ['Silhouette', 'Indizi', 'Attesa', 'Rivelazione'],
  kinetic_strips: ['Titolo', 'Pannelli', 'Slice', 'Accelerazione'],
  beat_montage: ['Intro', 'Tagli sul beat', 'Build-up', 'Drop finale'],
};

let analysisRequest;
let pollTimer;

function selectedStyle() {
  return form.querySelector('input[name="style"]:checked').value;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB';
  const megabytes = bytes / (1024 * 1024);
  return megabytes >= 1 ? `${megabytes.toFixed(megabytes >= 10 ? 0 : 1)} MB` : `${Math.ceil(bytes / 1024)} KB`;
}

function summarizeFiles(input, target, singular, plural) {
  const files = [...input.files];
  if (!files.length) {
    target.textContent = input === mediaInput ? 'Nessun contenuto selezionato.' : 'Nessuna canzone selezionata.';
    return;
  }

  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  const label = files.length === 1 ? singular : plural;
  const names = files.slice(0, 2).map((file) => file.name).join(', ');
  const remaining = files.length > 2 ? ` e altri ${files.length - 2}` : '';
  target.textContent = `${files.length} ${label} · ${formatBytes(totalSize)} · ${names}${remaining}`;
}

function updatePresetPhases() {
  presetPhaseList.replaceChildren();
  presetPhases[selectedStyle()].forEach((phase) => {
    const item = document.createElement('li');
    item.textContent = phase;
    presetPhaseList.append(item);
  });
}

async function responseError(response, fallback) {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    const body = await response.json().catch(() => ({}));
    return body.error || body.message || fallback;
  }
  if (response.status >= 500) {
    return 'Il server si sta avviando o è momentaneamente occupato. Attendi un minuto e riprova.';
  }
  return fallback;
}

function showAnalysisLoading() {
  analysisPanel.setAttribute('aria-busy', 'true');
  analysisEmpty.hidden = false;
  analysisEmpty.lastChild.textContent = ' Analisi del ritmo in corso…';
  analysisResult.hidden = true;
  analysisNote.hidden = true;
}

function showAnalysisError(message) {
  analysisPanel.setAttribute('aria-busy', 'false');
  analysisEmpty.hidden = false;
  analysisEmpty.lastChild.textContent = ` ${message}`;
  analysisResult.hidden = true;
  analysisNote.hidden = true;
}

function showAnalysis(data) {
  analysisPanel.setAttribute('aria-busy', 'false');
  analysisEmpty.hidden = true;
  analysisResult.hidden = false;
  musicBpm.textContent = data.bpm ?? '—';
  editBpm.textContent = data.edit_bpm ?? data.bpm ?? '—';
  contentCount.textContent = data.recommended_content ?? '—';
  visualCuts.textContent = data.visual_cuts ?? '—';

  const duration = Number(data.duration) || Number(durationSelect.value);
  const durationLabel = Number.isInteger(duration) ? String(duration) : duration.toFixed(1).replace('.', ',');
  const contents = data.recommended_content;
  analysisNote.textContent = contents
    ? `Per ${durationLabel} secondi di musica sono consigliati circa ${contents} contenuti distinti. Se ne carichi uno solo, PetCut lo riutilizzerà con inquadrature e movimenti diversi.`
    : (data.message || 'La canzone è pronta per guidare il montaggio.');
  analysisNote.hidden = false;
}

async function analyzeSong() {
  if (!audioInput.files[0]) return;
  if (analysisRequest) analysisRequest.abort();
  analysisRequest = new AbortController();
  showAnalysisLoading();

  const data = new FormData();
  data.append('audio', audioInput.files[0]);
  data.append('style', selectedStyle());
  data.append('duration', durationSelect.value);

  try {
    const response = await fetch('/api/analyze-audio', {
      method: 'POST',
      body: data,
      signal: analysisRequest.signal,
    });
    if (!response.ok) {
      throw new Error(await responseError(response, 'Non riesco ad analizzare questa canzone.'));
    }
    showAnalysis(await response.json());
  } catch (error) {
    if (error.name !== 'AbortError') {
      showAnalysisError(`${error.message} Puoi comunque generare il video.`);
    }
  }
}

function setProgress(value, stage, message) {
  const safeValue = Math.max(0, Math.min(100, Math.round(value)));
  renderProgress.value = safeValue;
  renderProgress.textContent = `${safeValue}%`;
  renderProgress.setAttribute('aria-valuenow', String(safeValue));
  progressPercent.textContent = `${safeValue}%`;
  if (message) renderStatus.textContent = message;

  renderSteps.forEach((item, index) => {
    item.classList.toggle('complete', index < stage);
    item.classList.toggle('active', index === stage);
  });
}

function progressForElapsed(seconds) {
  if (seconds < 12) {
    return { value: 8 + seconds * 1.1, stage: 0, message: 'Analisi di BPM, attacchi e struttura della canzone…' };
  }
  if (seconds < 42) {
    return { value: 21 + (seconds - 12) * 0.65, stage: 1, message: 'Preparazione di soggetti, ritagli e inquadrature…' };
  }
  if (seconds < 150) {
    return { value: 41 + (seconds - 42) * 0.37, stage: 2, message: 'Creazione di tagli, testi, movimenti e transizioni…' };
  }
  return { value: Math.min(94, 81 + (seconds - 150) * 0.08), stage: 3, message: 'Esportazione del video verticale…' };
}

function resetResult() {
  resultVideo.pause();
  resultVideo.removeAttribute('src');
  resultVideo.load();
  downloadButton.href = '#';
  resultPanel.hidden = true;
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function clearFormError() {
  formError.textContent = '';
  formError.hidden = true;
}

async function pollJob(jobId, startedAt) {
  while (true) {
    await new Promise((resolve) => {
      pollTimer = window.setTimeout(resolve, 2500);
    });

    const response = await fetch(`/api/render/${jobId}`);
    if (!response.ok) {
      throw new Error(await responseError(response, 'Non riesco a controllare lo stato del montaggio.'));
    }

    const job = await response.json();
    if (job.status === 'complete') return job;
    if (job.status !== 'processing') {
      throw new Error(job.error || 'Generazione non riuscita. Riprova con file più brevi.');
    }

    const elapsed = (Date.now() - startedAt) / 1000;
    const simulated = progressForElapsed(elapsed);
    const actualProgress = Number(job.progress);
    setProgress(
      Number.isFinite(actualProgress) ? actualProgress : simulated.value,
      Number.isFinite(Number(job.stage)) ? Number(job.stage) : simulated.stage,
      job.phase || simulated.message,
    );
  }
}

mediaInput.addEventListener('change', () => {
  summarizeFiles(mediaInput, mediaSummary, 'contenuto', 'contenuti');
});

audioInput.addEventListener('change', () => {
  summarizeFiles(audioInput, audioSummary, 'canzone', 'canzoni');
  analyzeSong();
});

durationSelect.addEventListener('change', analyzeSong);

form.querySelectorAll('input[name="style"]').forEach((radio) => {
  radio.addEventListener('change', () => {
    updatePresetPhases();
    analyzeSong();
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearFormError();

  if (!form.reportValidity()) return;

  window.clearTimeout(pollTimer);
  resetResult();
  submitButton.disabled = true;
  form.setAttribute('aria-busy', 'true');
  renderPanel.hidden = false;
  setProgress(4, 0, 'Caricamento e preparazione dei file…');
  renderPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  renderHeading.focus({ preventScroll: true });

  try {
    const response = await fetch('/api/render', {
      method: 'POST',
      body: new FormData(form),
    });
    if (!response.ok) {
      throw new Error(await responseError(response, 'Generazione non riuscita.'));
    }

    const { job_id: jobId } = await response.json();
    if (!jobId) throw new Error('Il server non ha restituito un identificativo del montaggio.');

    const job = await pollJob(jobId, Date.now());
    setProgress(100, 4, 'Video completato.');

    const downloadUrl = `/api/render/${jobId}/download`;
    resultVideo.src = downloadUrl;
    downloadButton.href = downloadUrl;

    const rhythm = job.edit_bpm && job.edit_bpm !== job.bpm
      ? `${job.bpm} BPM, montaggio a ${job.edit_bpm} BPM`
      : `${job.bpm || '—'} BPM`;
    resultSummary.textContent = `Ritmo: ${rhythm}. Cambi visivi creati: ${job.scenes || '—'}.`;
    resultPanel.hidden = false;
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    resultHeading.focus({ preventScroll: true });
  } catch (error) {
    setProgress(renderProgress.value, 3, 'Il montaggio si è interrotto.');
    showFormError(error.message);
    formError.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } finally {
    submitButton.disabled = false;
    form.removeAttribute('aria-busy');
  }
});

newEditButton.addEventListener('click', () => {
  resetResult();
  renderPanel.hidden = true;
  clearFormError();
  form.scrollIntoView({ behavior: 'smooth', block: 'start' });
  mediaInput.focus({ preventScroll: true });
});

updatePresetPhases();
