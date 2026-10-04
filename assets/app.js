const pages = [...document.querySelectorAll('.book-page')];
const sidebar = document.querySelector('#sidebar');
const menu = document.querySelector('#menu');
let navigation = 0;
let mathQueue = Promise.resolve();

async function navigate() {
  const current = ++navigation;
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { id = ''; }
  const target = id ? document.getElementById(id) : pages[0];
  const selected = target?.closest('.book-page') || pages[0];
  pages.forEach(page => { page.hidden = page !== selected; });
  // A cross-reference can point into a folded chapter. Reveal its target first.
  let ancestor = target?.parentElement;
  const opened = [];
  while (ancestor && ancestor !== selected) {
    if (ancestor.tagName === 'DETAILS') {
      ancestor.classList.remove('tex2jax_ignore');
      ancestor.open = true;
      opened.push(ancestor);
    }
    ancestor = ancestor.parentElement;
  }
  sidebar.querySelectorAll('a[href^="#"]').forEach(link => {
    const linkTarget = document.getElementById(decodeURIComponent(link.hash.slice(1)));
    const active = linkTarget === target || linkTarget === selected || linkTarget?.id === selected.dataset.chapter;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  document.title = selected.querySelector('h1').textContent.trim() + ' · SO & AI';
  sidebar.classList.remove('open');
  menu.setAttribute('aria-expanded', 'false');
  window.scrollTo(0, 0);
  mathQueue = mathQueue.then(async () => {
    if (window.MathJax?.startup?.promise) {
      await window.MathJax.startup.promise;
      if (current === navigation && !selected.dataset.typeset) {
        await window.MathJax.typesetPromise([selected]);
        selected.dataset.typeset = 'true';
      }
      for (const detail of opened) {
        if (!detail.dataset.typeset) {
          await window.MathJax.typesetPromise([detail]);
          detail.dataset.typeset = 'true';
        }
      }
    }
  }).catch(error => console.warn('Formula rendering:', error));
  await mathQueue;
  if (current !== navigation) return;
  if (target && target !== selected && selected.contains(target)) target.scrollIntoView();
}

menu.addEventListener('click', () => {
  const open = sidebar.classList.toggle('open');
  menu.setAttribute('aria-expanded', String(open));
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    sidebar.classList.remove('open');
    menu.setAttribute('aria-expanded', 'false');
    menu.focus();
  }
});
document.addEventListener('click', event => {
  const link = event.target.closest('a[href^="#"]');
  if (link && link.hash === location.hash) {
    event.preventDefault();
    navigate();
  }
});
window.addEventListener('hashchange', navigate);
window.addEventListener('DOMContentLoaded', navigate);

document.querySelectorAll('.original-text').forEach(detail => {
  detail.addEventListener('toggle', () => {
    if (!detail.open || detail.dataset.typeset) return;
    detail.classList.remove('tex2jax_ignore');
    mathQueue = mathQueue.then(async () => {
      if (!window.MathJax?.startup?.promise) return;
      await window.MathJax.startup.promise;
      await window.MathJax.typesetPromise([detail]);
      detail.dataset.typeset = 'true';
    }).catch(error => console.warn('Formula rendering:', error));
  });
});

document.querySelectorAll('.book-content pre').forEach(pre => {
  const wrapper = document.createElement('div');
  wrapper.className = 'code-wrap';
  pre.before(wrapper);
  wrapper.append(pre);
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.textContent = '复制';
  copy.setAttribute('aria-label', '复制代码');
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText((pre.querySelector('code') || pre).textContent);
      copy.textContent = '已复制';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(pre);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copy.textContent = '请按 Cmd/Ctrl+C';
    }
    setTimeout(() => { copy.textContent = '复制'; }, 2000);
  });
  wrapper.append(copy);
});
