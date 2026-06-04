import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  chessComArchivesUrl,
  chessComGameColor,
  clampMax,
  fetchChessComPgn,
  fetchLichessPgn,
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
