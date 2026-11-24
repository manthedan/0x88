#!/usr/bin/env node
/**
 * Render the link-preview image (public/social-card.png, 1200x630) and the
 * Apple touch icon (public/apple-touch-icon.png, 180x180).
 *
 * The board is captured from the landing page's real example board so the card
 * matches the site. Requires agent-browser.
 *
 *   node scripts/social-card/render.mjs [landing-url]   (default https://0x88.app/)
 */
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const landingUrl = process.argv[2] ?? 'https://0x88.app/';
const here = resolve(import.meta.dirname);
const repo = resolve(here, '../..');
const work = mkdtempSync(join(tmpdir(), 'social-card-'));
const env = { ...process.env, AGENT_BROWSER_SESSION: 'social-card' };
const browser = (...args) => execFileSync('agent-browser', args, { env, encoding: 'utf8' });

const card = join(work, 'social-card.png');
const icon = join(work, 'apple-touch-icon.png');
try {
  // 1. The example board, dark theme, at 2x so the card stays sharp.
  browser('set', 'viewport', '1440', '900', '2');
  browser('set', 'media', 'dark');
  browser('open', landingUrl);
  browser('wait', '.preview-board');
  browser('wait', '1500');
  browser('screenshot', '.preview-board', join(here, 'board.png'));

  // 2. The card itself.
  browser('set', 'viewport', '1200', '630', '2');
  browser('open', pathToFileURL(join(here, 'card.html')).href);
  browser('wait', '800');
  const card2x = join(work, 'card@2x.png');
  browser('screenshot', card2x);

  // 3. The touch icon: the favicon on a full square, since iOS masks the corners itself.
  browser('set', 'viewport', '180', '180', '1');
  browser('open', pathToFileURL(join(here, 'icon.html')).href);
  browser('wait', '500');
  browser('screenshot', icon);

  // Downsample the 2x capture for smooth text at the 1200x630 size platforms request.
  execFileSync('magick', [card2x, '-filter', 'Lanczos', '-resize', '1200x630!', '-strip', card]);

  copyFileSync(card, join(repo, 'public/social-card.png'));
  copyFileSync(icon, join(repo, 'public/apple-touch-icon.png'));
} finally {
  try {
    browser('close');
  } catch {}
  rmSync(work, { recursive: true, force: true });
  rmSync(join(here, 'board.png'), { force: true });
}

for (const file of ['public/social-card.png', 'public/apple-touch-icon.png']) {
  console.log(`${file} ${(statSync(join(repo, file)).size / 1024).toFixed(0)} KB`);
}
