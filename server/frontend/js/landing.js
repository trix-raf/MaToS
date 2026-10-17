/* MaToS — landing page behaviour: mobile menu, scroll-spy, FAQ accordion */
(function () {
  'use strict';

  /* ---- Mobile menu ---- */
  var nav = document.querySelector('.site-nav');
  var toggle = document.querySelector('.site-nav__toggle');
  if (nav && toggle) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(open));
    });
    nav.querySelectorAll('.site-nav__link').forEach(function (a) {
      a.addEventListener('click', function () {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* ---- Scroll-spy: highlight the nav link for the section in view ---- */
  var links = Array.prototype.slice.call(document.querySelectorAll('.site-nav__link[href^="#"]'));
  var targets = links
    .map(function (a) { return document.querySelector(a.getAttribute('href')); })
    .filter(Boolean);
  var footer = document.querySelector('.site-footer');
  var footerBasePadding = footer ? parseFloat(window.getComputedStyle(footer).paddingBottom) : 0;

  links.forEach(function (link) {
    link.addEventListener('click', function (event) {
      var target = document.querySelector(link.getAttribute('href'));
      if (!target) return;

      event.preventDefault();
      revealSection(target);
      var targetRect = target.getBoundingClientRect();
      var scrollTop = window.scrollY + targetRect.top + targetRect.height / 2 - window.innerHeight / 2;
      var needsExtraScroll = false;
      if (footer) {
        footer.style.paddingBottom = footerBasePadding + 'px';
        var maxScroll = document.documentElement.scrollHeight - window.innerHeight;
        var extraPadding = Math.max(0, Math.ceil(scrollTop - maxScroll));
        if (extraPadding > 0) {
          footer.style.paddingBottom = footerBasePadding + extraPadding + 'px';
          needsExtraScroll = true;
        }
      }
      var hash = link.getAttribute('href');
      if (window.location.hash !== hash) window.history.pushState(null, '', hash);

      var behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
      var scrollToTarget = function () { window.scrollTo({ top: scrollTop, behavior: behavior }); };
      if (needsExtraScroll) requestAnimationFrame(scrollToTarget);
      else scrollToTarget();
    });
  });

  targets.forEach(function (section) {
    section.querySelectorAll('h1, h2, h3, p, .pillar__value, .pillar__label').forEach(function (item, index) {
      item.classList.add('section-reveal', index % 2 === 0 ? 'section-reveal--rise' : 'section-reveal--slide');
      item.style.setProperty('--reveal-delay', Math.min(index, 5) * 70 + 'ms');
    });
  });

  function revealSection(section) {
    section.querySelectorAll('.section-reveal').forEach(function (item) {
      item.classList.add('is-visible');
    });
  }

  function setActive(id) {
    links.forEach(function (a) { a.classList.toggle('is-active', a.getAttribute('href') === '#' + id); });
  }

  function onScroll() {
    var y = window.scrollY + window.innerHeight / 2;
    var current = null;
    targets.forEach(function (t) { if (t.offsetTop <= y) current = t; });
    setActive(current ? current.id : '');
    if (current) revealSection(current);
  }

  var ticking = false;
  window.addEventListener('scroll', function () {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () { onScroll(); ticking = false; });
  }, { passive: true });
  onScroll();

  /* ---- FAQ accordion ---- */
  document.querySelectorAll('.faq-item__q').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var item = btn.closest('.faq-item');
      var open = !item.classList.contains('is-open');
      item.classList.toggle('is-open', open);
      btn.setAttribute('aria-expanded', String(open));
    });
  });
})();
