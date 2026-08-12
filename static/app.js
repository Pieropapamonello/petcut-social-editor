const form = document.querySelector('#editor-form');
const button = form.querySelector('button');
const status = document.querySelector('#status');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  status.textContent = 'Analizzo il beat e genero il tuo edit: potrebbe volerci qualche minuto…';
  try {
    const response = await fetch('/api/render', { method: 'POST', body: new FormData(form) });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'Generazione non riuscita.');
    const blob = await response.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'petcut-social-edit.mp4';
    link.click();
    URL.revokeObjectURL(link.href);
    status.textContent = `Pronto! BPM rilevato: ${response.headers.get('X-Detected-BPM') || '—'}. Il download è iniziato.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
