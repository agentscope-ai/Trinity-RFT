/* A parameter-count illustration, not a model-quality or runtime simulation. */
document.querySelectorAll('[data-lora-rank-demo]').forEach(function (figure) {
  const dimension = 4096;
  const buttons = Array.from(figure.querySelectorAll('[data-rank]'));
  function select(rank) {
    if (![8, 16, 32, 64, 128].includes(rank)) return;
    buttons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.rank) === rank)));
    figure.querySelectorAll('[data-rank-value]').forEach(node => { node.textContent = rank; });
    const count = 2 * dimension * rank;
    figure.querySelector('[data-parameter-count]').textContent = count.toLocaleString('en-US');
    figure.querySelector('[data-parameter-ratio]').textContent = (100 * count / (dimension * dimension)).toFixed(2) + '%';
    figure.style.setProperty('--rank-width', (8 + rank / 8) + 'px');
  }
  buttons.forEach(button => button.addEventListener('click', () => select(Number(button.dataset.rank))));
  select(32);
});
