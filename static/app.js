const form = document.querySelector('#editor-form');
const button = form.querySelector('button');
const status = document.querySelector('#status');
const analysis = document.querySelector('#song-analysis');
const audioInput = form.elements.audio;

async function analyzeSong() {
  if (!audioInput.files[0]) return;
  analysis.textContent = 'Analisi del ritmo in corso…';
  const data = new FormData();
  data.append('audio', audioInput.files[0]);
  data.append('style', form.elements.style.value);
  data.append('duration', form.elements.duration.value);
  try {
    const response = await fetch('/api/analyze-audio', { method: 'POST', body: data });
    if (!response.ok) throw new Error();
    analysis.textContent = (await response.json()).message;
  } catch {
    analysis.textContent = 'Non riesco ad analizzare questa canzone. Puoi comunque generare l’edit.';
  }
}
audioInput.addEventListener('change', analyzeSong);
form.elements.style.addEventListener('change', analyzeSong);
form.elements.duration.addEventListener('change', analyzeSong);

form.addEventListener('submit', async (event) => {
  event.preventDefault(); button.disabled = true;
  status.textContent = 'Preparazione dell’edit…';
  try {
    const response = await fetch('/api/render', { method: 'POST', body: new FormData(form) });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'Generazione non riuscita.');
    const { job_id: jobId } = await response.json();
    let job;
    do {
      await new Promise((resolve) => setTimeout(resolve, 2500));
      const progress = await fetch(`/api/render/${jobId}`);
      job = await progress.json();
      status.textContent = job.status === 'complete' ? 'Video pronto: avvio il download…' : 'Montaggio in corso: puoi lasciare aperta questa pagina.';
    } while (job.status === 'processing');
    if (job.status !== 'complete') throw new Error(job.error || 'Generazione non riuscita.');
    const download = await fetch(`/api/render/${jobId}/download`);
    if (!download.ok) throw new Error('Il download non è disponibile. Riprova.');
    const blob = await download.blob(); const link = document.createElement('a');
    link.href = URL.createObjectURL(blob); link.download = 'petcut-social-edit.mp4'; link.click(); URL.revokeObjectURL(link.href);
    status.textContent = `Pronto! BPM: ${job.bpm || '—'}; scene ritmiche create: ${job.scenes || '—'}. Il download è iniziato.`;
  } catch (error) { status.textContent = error.message; } finally { button.disabled = false; }
});
