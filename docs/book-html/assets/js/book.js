// ============================================
//   ML Book — Main JS
// ============================================

document.addEventListener('DOMContentLoaded', () => {

  // ---- Code copy buttons ----
  document.querySelectorAll('pre').forEach(pre => {
    const btn = pre.querySelector('.code-copy');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      navigator.clipboard.writeText(code.innerText).then(() => {
        btn.textContent = 'Скопировано!';
        setTimeout(() => btn.textContent = 'Копировать', 2000);
      });
    });
  });

  // ---- Active nav item ----
  const currentPage = window.location.pathname.split('/').pop();
  document.querySelectorAll('.nav-item').forEach(link => {
    const href = link.getAttribute('href');
    if (href && href.includes(currentPage)) {
      link.classList.add('active');
    }
  });

  // ---- Reading progress bar ----
  const progressFill = document.querySelector('.progress-bar-fill');
  if (progressFill) {
    const updateProgress = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      progressFill.style.width = Math.min(progress, 100) + '%';
    };
    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  // ---- Smooth scroll for anchor links ----
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ---- Highlight active section in TOC ----
  const headings = document.querySelectorAll('h2[id], h3[id]');
  const tocLinks = document.querySelectorAll('.toc-link');

  if (headings.length && tocLinks.length) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          tocLinks.forEach(link => {
            link.classList.toggle(
              'active',
              link.getAttribute('href') === '#' + entry.target.id
            );
          });
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });

    headings.forEach(h => observer.observe(h));
  }

});

// ============================================
//   Chatbot placeholder (будет добавлен позже)
// ============================================
// TODO: floating chat widget with text selection support
