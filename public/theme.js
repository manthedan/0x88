/* Shared theme toggle + board style logic. Loaded after DOM ready.
   The early inline script in <head> applies the saved theme before
   first paint to prevent FOUC; this file wires up the toggle button
   and board-style selector. */
(function () {
  var root = document.documentElement;
  var boards = ['brown', 'night', 'blue', 'green'];

  function isDark() {
    return root.dataset.theme === 'dark' ||
      (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  }

  function sync() {
    document.querySelectorAll('#themeToggle').forEach(function (button) {
      var dark = isDark();
      button.textContent = dark ? '\u2600' : '\u263E';
      button.setAttribute('aria-pressed', String(dark));
      button.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    });

    document.querySelectorAll('#boardStyleSelect').forEach(function (select) {
      var current = root.dataset.board || 'brown';
      select.value = boards.indexOf(current) >= 0 ? current : 'brown';
    });
  }

  window.__0x88SyncThemeControls = sync;

  if (!window.__0x88ThemeControlsInstalled) {
    window.__0x88ThemeControlsInstalled = true;
    document.addEventListener('click', function (event) {
      var button = event.target && event.target.closest && event.target.closest('#themeToggle');
      if (!button) return;
      root.dataset.theme = isDark() ? 'light' : 'dark';
      try { localStorage.setItem('0x88-theme', root.dataset.theme); } catch (e) {}
      sync();
    });

    document.addEventListener('change', function (event) {
      var select = event.target && event.target.closest && event.target.closest('#boardStyleSelect');
      if (!select) return;
      var val = select.value;
      if (val === 'brown') delete root.dataset.board;
      else root.dataset.board = val;
      try { localStorage.setItem('0x88-board', val); } catch (e) {}
      sync();
    });

    try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', sync); } catch (e) {}
  }

  sync();
})();
