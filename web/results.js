'use strict';
const thumbnails = [...document.querySelectorAll('.result-workspace .thumbnail')];
function selectView(button) {
  const image = document.querySelector('#result-image');
  image.src = button.dataset.url;
  image.alt = button.dataset.caption;
  document.querySelector('#result-image-link').href = button.dataset.url;
  document.querySelector('#result-caption').textContent = button.dataset.caption;
  for (const other of thumbnails) {
    other.classList.toggle('selected', other === button);
    other.setAttribute('aria-pressed', String(other === button));
  }
}
for (const button of thumbnails) button.addEventListener('click', () => selectView(button));
for (const filter of document.querySelectorAll('.result-workspace [data-filter]')) {
  filter.addEventListener('click', () => {
    for (const other of document.querySelectorAll('.result-workspace [data-filter]')) {
      other.classList.toggle('selected', other === filter);
      other.setAttribute('aria-pressed', String(other === filter));
    }
    for (const button of thumbnails) button.hidden = filter.dataset.filter !== 'all' && button.dataset.kind !== filter.dataset.filter;
    if (!thumbnails.some(button => !button.hidden && button.classList.contains('selected'))) {
      const first = thumbnails.find(button => !button.hidden);
      if (first) selectView(first);
    }
  });
}
