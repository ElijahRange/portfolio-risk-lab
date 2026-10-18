// Portfolio assistant (views/chat.py): small DOM helpers the Dash callbacks can't do.
// - Escape closes the chat.
// - A chat link to a portfolio tab (/?tab=risk) also switches tabs when the URL is already that one.
// - On phones, where the chat covers the page, following a link closes it.
document.addEventListener('keydown', function (e) {
  if (e.key !== 'Escape') { return; }
  // The Edit portfolio dialog sits above the chat, so close it first.
  var target = document.querySelector('.pe-overlay.open') ? 'pe-close'
    : (document.querySelector('.chat-panel.open') ? 'chat-close' : null);
  var button = target && document.getElementById(target);
  if (button) { button.click(); }
});

document.addEventListener('click', function (e) {
  var link = e.target.closest ? e.target.closest('.chat-link') : null;
  if (!link) { return; }
  var wrap = link.closest('[data-tab]');
  if (wrap) {
    var label = wrap.getAttribute('data-tab');
    setTimeout(function () {
      document.querySelectorAll('#tabs .tab').forEach(function (t) {
        if (t.textContent.trim() === label && !t.classList.contains('tab--active')) { t.click(); }
      });
    }, 500);
  }
  if (window.innerWidth <= 760) {
    var close = document.getElementById('chat-close');
    if (close) { setTimeout(function () { close.click(); }, 0); }
  }
});
