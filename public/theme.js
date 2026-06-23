/* Shared theme toggle + board style logic. Loaded after DOM ready.
   The early inline script in <head> applies the saved theme before
   first paint to prevent FOUC; this file wires up the toggle button
   and board-style selector. */
(function () {
  var root = document.documentElement;
  var button = document.getElementById('themeToggle');
  if (button) {
    function sync() {
      var dark = root.dataset.theme === 'dark' ||
        (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
      button.textContent = dark ? '\u2600' : '\u263E';
      button.setAttribute('aria-pressed', String(dark));
      button.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    }

    button.addEventListener('click', function () {
      var isDark = root.dataset.theme === 'dark' ||
        (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
      if (isDark) {
        root.dataset.theme = 'light';
      } else {
        root.dataset.theme = 'dark';
      }
      try { localStorage.setItem('0x88-theme', root.dataset.theme); } catch (e) {}
      sync();
    });

    sync();
  }

  var boardSelect = document.getElementById('boardStyleSelect');
  if (boardSelect) {
    var boards = ['brown', 'night', 'blue', 'green'];
    var current = root.dataset.board || 'brown';
    boardSelect.value = boards.indexOf(current) >= 0 ? current : 'brown';
    boardSelect.addEventListener('change', function () {
      var val = boardSelect.value;
      if (val === 'brown') delete root.dataset.board;
      else root.dataset.board = val;
      try { localStorage.setItem('0x88-board', val); } catch (e) {}
    });
  }
})();
