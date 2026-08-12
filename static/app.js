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
  status.textContent = 'Analizzo il beat e genero il tuo edit: potrebbe volerci qualche minuto…';
  try {
    const response = await fetch('/api/render', { method: 'POST', body: new FormData(form) });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'Generazione non riuscita.');
    const blob = await response.blob(); const link = document.createElement('a');
    link.href = URL.createObjectURL(blob); link.download = 'petcut-social-edit.mp4'; link.click(); URL.revokeObjectURL(link.href);
    status.textContent = `Pronto! BPM: ${response.headers.get('X-Detected-BPM') || '—'}; scene ritmiche create: ${response.headers.get('X-Recommended-Content') || '—'}. Il download è iniziato.`;
  } catch (error) { status.textContent = error.message; } finally { button.disabled = false; }
});
