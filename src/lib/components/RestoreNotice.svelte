<script lang="ts">
export let area: 'analysis' | 'play';
</script>

<div id={`${area}RestoreNotice`} class="restore-notice" hidden>
  <span>Restored your last game</span>
  <button id={`${area}RestoreNew`} type="button">Start new</button>
  <button id={`${area}RestoreDismiss`} class="dismiss" type="button" aria-label="Dismiss restore notice">×</button>
</div>

<style>
.restore-notice { display:flex; flex-wrap:wrap; align-items:center; gap:6px 10px; padding:10px 12px; margin-bottom:10px; border:1px solid var(--rule); border-radius:var(--radius-sm); background:var(--panel-inset); color:var(--text-soft); font:13px/1.4 var(--sans); }
.restore-notice[hidden] { display:none; }
.restore-notice button { padding:4px 6px; border:0; background:transparent; color:var(--accent); text-decoration:underline; font:inherit; }
.restore-notice .dismiss { margin-left:auto; text-decoration:none; color:var(--muted); font-size:20px; line-height:1; }
</style>
