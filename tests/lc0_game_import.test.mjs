import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  chessComArchivesUrl,
  chessComGameColor,
  clampMax,
  fetchChessComPgn,
  fetchGameHistoryPgn,
  fetchLichessPgn,
  IMPORT_REQUEST_TIMEOUT_MS,
  lichessGamesUrl,
  selectChessComPgns,
} from '../src/lc0/gameImport.ts';

test('clampMax validates and bounds the count', () => {
  assert.equal(clampMax(undefined), 50);
  assert.equal(clampMax(0), 50);
  assert.equal(clampMax(-5), 50);
  assert.equal(clampMax(10), 10);
  assert.equal(clampMax(99999), 300);
});

test('lichessGamesUrl encodes user and options', () => {
  const url = lichessGamesUrl('Magnus Carlsen', { max: 20, color: 'white', rated: true });
  assert.match(url, /lichess\.org\/api\/games\/user\/Magnus%20Carlsen\?/);
  assert.match(url, /max=20/);
  assert.match(url, /color=white/);
  assert.match(url, /rated=true/);
  // 'both' rated omits the param
  assert.doesNotMatch(lichessGamesUrl('x', { rated: '' }), /rated=/);
});

test('chessComArchivesUrl lowercases the username', () => {
  assert.equal(chessComArchivesUrl('Hikaru'), 'https://api.chess.com/pub/player/hikaru/games/archives');
});

test('chessComGameColor identifies the player side', () => {
  const game = { white: { username: 'Hikaru' }, black: { username: 'Opponent' } };
  assert.equal(chessComGameColor(game, 'hikaru'), 'white');
  assert.equal(chessComGameColor(game, 'opponent'), 'black');
  assert.equal(chessComGameColor(game, 'someone'), undefined);
});

test('selectChessComPgns takes most-recent games with a color filter', () => {
  const games = [
    { pgn: 'g1', white: { username: 'me' }, black: { username: 'a' } },
    { pgn: 'g2', white: { username: 'b' }, black: { username: 'me' } },
    { pgn: 'g3', white: { username: 'me' }, black: { username: 'c' } },
  ];
  assert.deepEqual(selectChessComPgns(games, 'me', { max: 2 }), ['g3', 'g2'], 'newest first');
  assert.deepEqual(selectChessComPgns(games, 'me', { color: 'white' }), ['g3', 'g1']);
});

test('fetchLichessPgn returns PGN text and maps 404', async () => {
  const okFetch = async () => ({ ok: true, status: 200, text: async () => '[Event "x"]\n\n1. e4 *', json: async () => ({}) });
  assert.match(await fetchLichessPgn('user', { max: 1 }, okFetch), /1\. e4/);
  const notFound = async () => ({ ok: false, status: 404, text: async () => '', json: async () => ({}) });
  await assert.rejects(() => fetchLichessPgn('nope', {}, notFound), /not found/);
});

test('fetchChessComPgn walks archives newest-first until max', async () => {
  const calls = [];
  const fakeFetch = async (url) => {
    calls.push(url);
    if (url.endsWith('/archives')) {
      return { ok: true, status: 200, json: async () => ({ archives: ['url/2024/01', 'url/2024/02'] }), text: async () => '' };
    }
    if (url.endsWith('2024/02')) {
      return { ok: true, status: 200, json: async () => ({ games: [{ pgn: 'feb1', white: { username: 'me' } }] }), text: async () => '' };
    }
    return { ok: true, status: 200, json: async () => ({ games: [{ pgn: 'jan1', white: { username: 'me' } }] }), text: async () => '' };
  };
  const pgn = await fetchChessComPgn('me', { max: 2 }, fakeFetch);
  assert.equal(pgn, 'feb1\n\njan1', 'newest month first');
  assert.equal(calls[0], 'https://api.chess.com/pub/player/me/games/archives');
});

// A fetch that never answers until its signal aborts, like a stalled connection.
function stalledFetch(seen = []) {
  return (url, init) => {
    seen.push({ url, signal: init?.signal });
    return new Promise((_, reject) => {
      const signal = init?.signal;
      if (!signal) return;
      signal.addEventListener('abort', () => reject(signal.reason), { once: true });
    });
  };
}

test('a stalled import times out with a readable error', async () => {
  assert.ok(IMPORT_REQUEST_TIMEOUT_MS >= 30_000, 'the default leaves room for a 300-game Lichess export');
  const seen = [];
  await assert.rejects(() => fetchGameHistoryPgn('lichess', 'user', { timeoutMs: 30 }, stalledFetch(seen)), /Lichess did not respond within/);
  assert.ok(seen[0].signal instanceof AbortSignal, 'every request carries a signal');
  await assert.rejects(() => fetchGameHistoryPgn('chesscom', 'user', { timeoutMs: 30 }, stalledFetch()), /Chess\.com did not respond within/);
});

test('the timeout covers each Chess.com month, not only the archive list', async () => {
  const fakeFetch = async (url, init) => {
    if (url.endsWith('/archives')) return { ok: true, status: 200, json: async () => ({ archives: ['url/2024/01'] }), text: async () => '' };
    return stalledFetch()(url, init);
  };
  await assert.rejects(() => fetchChessComPgn('me', { timeoutMs: 30 }, fakeFetch), /Chess\.com did not respond within/);
});

test('the caller can cancel an import in flight, and the abort reason propagates', async () => {
  const controller = new AbortController();
  const pending = fetchGameHistoryPgn('lichess', 'user', {}, stalledFetch(), controller.signal);
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
});

test('an already-cancelled import makes no request', async () => {
  const seen = [];
  await assert.rejects(() => fetchGameHistoryPgn('chesscom', 'user', {}, stalledFetch(seen), AbortSignal.abort()), { name: 'AbortError' });
  assert.equal(seen.length, 0);
});

test('cancelling between Chess.com months stops the walk', async () => {
  const controller = new AbortController();
  const calls = [];
  const fakeFetch = async (url) => {
    calls.push(url);
    if (url.endsWith('/archives')) return { ok: true, status: 200, json: async () => ({ archives: ['url/2024/01', 'url/2024/02'] }), text: async () => '' };
    controller.abort();
    return { ok: true, status: 200, json: async () => ({ games: [] }), text: async () => '' };
  };
  await assert.rejects(() => fetchChessComPgn('me', { max: 5 }, fakeFetch, controller.signal), { name: 'AbortError' });
  assert.deepEqual(calls, ['https://api.chess.com/pub/player/me/games/archives', 'url/2024/02']);
});
