const pages = [...document.querySelectorAll('.book-page')];
const sidebar = document.querySelector('#sidebar');
const menu = document.querySelector('#menu');
let navigation = 0;
let mathQueue = Promise.resolve();
let language = 'zh';
const messages = {
  zh: {copy: '复制', copied: '已复制', copyLabel: '复制代码', manualCopy: '请按 Cmd/Ctrl+C'},
  en: {copy: 'Copy', copied: 'Copied', copyLabel: 'Copy code', manualCopy: 'Press Cmd/Ctrl+C'},
};

function setLanguage(next, announce = false) {
  language = next === 'en' ? 'en' : 'zh';
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-zh][data-en]').forEach(element => {
    element.textContent = element.getAttribute('data-' + language);
    element.lang = document.documentElement.lang;
  });
  for (const attribute of ['aria-label', 'alt', 'src', 'href']) {
    document.querySelectorAll('[data-' + attribute + '-zh][data-' + attribute + '-en]').forEach(element => {
      element.setAttribute(attribute, element.getAttribute('data-' + attribute + '-' + language));
    });
  }
  document.querySelectorAll('.language-switch button[data-language]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.language === language));
  });
  document.querySelectorAll('[data-language-view]').forEach(view => {
    view.hidden = view.dataset.languageView !== language;
  });
  document.querySelectorAll('.code-wrap>button').forEach(button => {
    button.textContent = messages[language].copy;
    button.setAttribute('aria-label', messages[language].copyLabel);
  });
  const selected = pages.find(page => !page.hidden);
  if (selected) document.title = selected.querySelector('h1').textContent.trim() + ' · SO & AI';
  try { localStorage.setItem('so-ai-language', language); } catch { /* Reading also works without storage. */ }
  if (announce) {
    const url = new URL(location.href);
    url.searchParams.set('lang', language);
    try { history.replaceState(null, '', url); } catch { /* Some file viewers restrict history. */ }
    document.querySelector('#language-status').textContent = language === 'zh' ? '已切换为中文' : 'Switched to English';
  }
}

let initialLanguage = new URLSearchParams(location.search).get('lang');
if (!['zh', 'en'].includes(initialLanguage)) {
  try { initialLanguage = localStorage.getItem('so-ai-language'); } catch { /* Use Chinese by default. */ }
}
setLanguage(initialLanguage);
document.querySelectorAll('.language-switch button[data-language]').forEach(button => {
  button.addEventListener('click', () => setLanguage(button.dataset.language, true));
});

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
  copy.textContent = messages[language].copy;
  copy.setAttribute('aria-label', messages[language].copyLabel);
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText((pre.querySelector('code') || pre).textContent);
      copy.textContent = messages[language].copied;
    } catch {
      const range = document.createRange();
      range.selectNodeContents(pre);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copy.textContent = messages[language].manualCopy;
    }
    setTimeout(() => { copy.textContent = messages[language].copy; }, 2000);
  });
  wrapper.append(copy);
});
