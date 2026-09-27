// Compteur de mots en direct sous la lettre : même découpage que
// `Letter.word_count` (espaces), pour que le chiffre affiché soit celui que la
// relecture et les réserves utiliseront.
document.addEventListener("input", (event) => {
  const area = event.target.closest("textarea[data-wordcount]");
  if (!area) return;
  const out = document.querySelector(area.dataset.wordcount);
  if (!out) return;
  const words = area.value.split(/\s+/).filter(Boolean).length;
  const min = Number(out.dataset.min);
  const max = Number(out.dataset.max);
  out.textContent = `${words} mots (attendu ${min}–${max})`;
  out.classList.toggle("warn", words < min || words > max);
});
