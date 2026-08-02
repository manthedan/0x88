import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as v from 'valibot';
import { ChessComArchivesResponseSchema, ChessComGamesResponseSchema } from '../src/lc0/gameImportSchema.ts';
import { FenInputFixtureListSchema, HistoryInputFixtureListSchema } from '../src/lc0/inputFixtureSchema.ts';
import { Lc0ArtifactChannelManifestSchema, Lc0ArtifactReleaseManifestSchema, Lc0ModelManifestSchema } from '../src/lc0/modelManifestSchema.ts';
import { WholeOnnxRuntimeManifestSchema } from '../src/lc0/wholeOnnxRuntimeManifestSchema.ts';
import { SquareFormerMetaSchema } from '../src/nn/squareFormerMetaSchema.ts';
import { TvmjsManifestSchema } from '../src/nn/tvmjsManifestSchema.ts';

function assertValiError(fn, fieldPattern) {
  try {
    fn();
  } catch (error) {
    assert.ok(error instanceof v.ValiError, `expected ValiError, got ${error}`);
    const summary = JSON.stringify(error.issues);
    assert.ok(fieldPattern.test(summary), `expected an issue at path matching ${fieldPattern}; got ${summary}`);
    return;
  }
  assert.fail('expected v.parse to throw a ValiError');
}

// --- Lc0ModelManifest (shape from public/models/lc0/manifest.json) ---

test('model manifest parses a realistic payload with extra fields', () => {
  const payload = {
    generatedBy: 'scripts/lc0_prepare_model_assets.mjs',
    note: 'Local LC0 browser model assets.',
    models: [
      {
        file: 't1-256x10-distilled-swa-2432500.batch1.f32.onnx',
        url: '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx',
        mode: 'symlink',
        source: '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx',
        bytes: 80895900,
        sha256: '9942420798729cfd3ba0f20ced7b1e9650dadc832527a5ad28f8b990ca0d230e',
      },
    ],
  };
  const parsed = v.parse(Lc0ModelManifestSchema, payload);
  assert.deepEqual(parsed, payload, 'loose schema preserves unknown fields');
});

test('model manifest tolerates a minimal payload', () => {
  assert.deepEqual(v.parse(Lc0ModelManifestSchema, {}), {});
  assert.deepEqual(v.parse(Lc0ModelManifestSchema, { models: [] }), { models: [] });
});

test('model manifest rejects entries missing required fields or with wrong types', () => {
  assertValiError(() => v.parse(Lc0ModelManifestSchema, { models: [{ url: '/m.onnx' }] }), /file/);
  assertValiError(() => v.parse(Lc0ModelManifestSchema, { models: [{ file: 'm.onnx' }] }), /url/);
  assertValiError(() => v.parse(Lc0ModelManifestSchema, { models: [{ file: 'm.onnx', url: '/m.onnx', bytes: '80895900' }] }), /bytes/);
  assertValiError(() => v.parse(Lc0ModelManifestSchema, { models: 'not-an-array' }), /models/);
});

// --- Lc0ArtifactChannelManifest (shape from scripts/write_artifact_release_manifests.mjs) ---

test('artifact channel manifest parses realistic v1/v2 payloads with extra fields', () => {
  const v2 = {
    schema: 'lc0_browser.artifact_channel_manifest.v2',
    channel: 'stable',
    releaseId: '2026-08-01-abcd1234ef56',
    releaseManifestUrl: '/releases/2026-08-01-abcd1234ef56.json',
    generatedAt: '2026-08-01T00:00:00.000Z',
  };
  assert.deepEqual(v.parse(Lc0ArtifactChannelManifestSchema, v2), v2);
  const v1 = { schema: 'lc0_browser.artifact_channel_manifest.v1', releaseUrl: '/releases/stable.json' };
  assert.deepEqual(v.parse(Lc0ArtifactChannelManifestSchema, v1), v1);
});

test('artifact channel manifest rejects wrong primitive types', () => {
  assertValiError(() => v.parse(Lc0ArtifactChannelManifestSchema, { releaseManifestUrl: 42 }), /releaseManifestUrl/);
  assertValiError(() => v.parse(Lc0ArtifactChannelManifestSchema, { releaseUrl: ['x'] }), /releaseUrl/);
});

// --- Lc0ArtifactReleaseManifest ---

test('artifact release manifest parses v1 artifactUrl entries', () => {
  const payload = {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    artifacts: [
      {
        logicalUrl: '/models/lc0/test.onnx',
        artifactUrl: '/artifacts/sha256/ba78/test.onnx',
        bytes: 3,
        sha256: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
        unexpectedFutureField: { nested: true },
      },
    ],
  };
  assert.deepEqual(v.parse(Lc0ArtifactReleaseManifestSchema, payload), payload);
});

test('artifact release manifest parses v2 raw/representation entries', () => {
  const payload = {
    schema: 'lc0-webgpu.artifact-release.v2',
    artifacts: [
      {
        name: 'test-model',
        file: 'test.onnx',
        raw: { bytes: 3, sha256: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad' },
        representations: [
          {
            encoding: 'identity',
            url: '/artifacts/sha256/ba78/identity',
            bytes: 3,
            sha256: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
          },
          { encoding: 'br', url: '/artifacts/sha256/ba78/br/bbbb', bytes: 2, sha256: 'b'.repeat(64) },
        ],
      },
    ],
  };
  assert.deepEqual(v.parse(Lc0ArtifactReleaseManifestSchema, payload), payload);
});

test('artifact release manifest rejects malformed entries', () => {
  const badEncoding = { artifacts: [{ representations: [{ encoding: 'gzip', url: '/x', bytes: 1, sha256: 'a' }] }] };
  assertValiError(() => v.parse(Lc0ArtifactReleaseManifestSchema, badEncoding), /encoding/);
  const missingRawSha = { artifacts: [{ raw: { bytes: 3 } }] };
  assertValiError(() => v.parse(Lc0ArtifactReleaseManifestSchema, missingRawSha), /sha256/);
  const wrongBytes = { artifacts: [{ logicalUrl: '/m.onnx', bytes: '3' }] };
  assertValiError(() => v.parse(Lc0ArtifactReleaseManifestSchema, wrongBytes), /bytes/);
});

// --- Chess.com PubAPI responses ---

test('chess.com archives response parses a realistic payload', () => {
  const payload = { archives: ['https://api.chess.com/pub/player/hikaru/games/2024/01', 'https://api.chess.com/pub/player/hikaru/games/2024/02'] };
  assert.deepEqual(v.parse(ChessComArchivesResponseSchema, payload), payload);
  assert.deepEqual(v.parse(ChessComArchivesResponseSchema, {}), {}, 'missing archives key is tolerated');
});

test('chess.com archives response rejects non-string-array archives', () => {
  assertValiError(() => v.parse(ChessComArchivesResponseSchema, { archives: 'https://api.chess.com/x' }), /archives/);
  assertValiError(() => v.parse(ChessComArchivesResponseSchema, { archives: [42] }), /archives/);
});

test('chess.com games response parses a realistic monthly archive payload', () => {
  const payload = {
    games: [
      {
        url: 'https://www.chess.com/game/live/123456789',
        pgn: '[Event "Live Chess"]\n\n1. e4 e5 2. Nf3 *',
        time_control: '600',
        end_time: 1704067200,
        rated: true,
        tcn: 'abcdefghijklmnopqrst',
        uuid: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        initial_setup: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        fen: 'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKBNR b KQkq - 1 2',
        time_class: 'rapid',
        rules: 'chess',
        eco: 'https://www.chess.com/openings/Kings-Pawn-Opening-1.e4-e5-2.Nf3',
        white: { username: 'Hikaru', rating: 3200, result: 'win', '@id': 'https://api.chess.com/pub/player/hikaru', uuid: 'uuid-white' },
        black: { username: 'Opponent', rating: 3100, result: 'checkmated', '@id': 'https://api.chess.com/pub/player/opponent', uuid: 'uuid-black' },
      },
    ],
  };
  const parsed = v.parse(ChessComGamesResponseSchema, payload);
  assert.deepEqual(parsed, payload, 'loose schema preserves the many unconsumed fields');
  assert.equal(parsed.games[0].white.username, 'Hikaru');
});

test('chess.com games response rejects wrong primitive types', () => {
  assertValiError(() => v.parse(ChessComGamesResponseSchema, { games: [{ pgn: 123 }] }), /pgn/);
  assertValiError(() => v.parse(ChessComGamesResponseSchema, { games: [{ white: 'Hikaru' }] }), /white/);
  assertValiError(() => v.parse(ChessComGamesResponseSchema, { games: [{ white: { username: 7 } }] }), /username/);
  assertValiError(() => v.parse(ChessComGamesResponseSchema, { games: 'none' }), /games/);
});

// --- TvmjsManifest (shape from public/runtimes/centipawn-tvmjs-webgpu/.../manifest.json) ---

function tvmjsManifestPayload() {
  return {
    schema: 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1',
    modelFamily: 'bt4-soap-rem-c19000-final',
    dtype: 'f32',
    version: 'v1',
    target: 'webgpu',
    hostTarget: { kind: 'llvm', mtriple: 'wasm32-unknown-unknown-wasm' },
    generatedAt: '2026-06-10T00:39:00.430Z',
    requiredFeatures: ['webgpu'],
    runtime: {
      tvmjsBundle: 'tvmjs.bundle.js',
      tvmjsRuntimeWasm: 'tvmjs_runtime.wasm',
      note: 'TVMJS/WebGPU whole-model export bundle.',
    },
    compilerProvenance: { gitCommit: '15b1d983938403e8f608e934e70c3693a9cf7d0a' },
    models: [{ batch: 16, wasm: 'model.batch16.tvmjs.wasm', probe: 'model.batch16.json', bytes: 123456, sha256: 'a'.repeat(64) }],
    files: [
      { path: 'tvmjs.bundle.js', bytes: 654321, sha256: 'b'.repeat(64) },
      { path: 'model.batch16.tvmjs.wasm', bytes: 123456, sha256: 'a'.repeat(64) },
    ],
  };
}

test('tvmjs manifest parses a realistic payload with extra fields', () => {
  const payload = tvmjsManifestPayload();
  assert.deepEqual(v.parse(TvmjsManifestSchema, payload), payload);
});

test('tvmjs manifest rejects missing required fields and wrong primitive types', () => {
  const noSchema = tvmjsManifestPayload();
  delete noSchema.schema;
  assertValiError(() => v.parse(TvmjsManifestSchema, noSchema), /schema/);
  const noTarget = tvmjsManifestPayload();
  delete noTarget.target;
  assertValiError(() => v.parse(TvmjsManifestSchema, noTarget), /target/);
  const badBatch = tvmjsManifestPayload();
  badBatch.models = [{ batch: '16', wasm: 'model.wasm' }];
  assertValiError(() => v.parse(TvmjsManifestSchema, badBatch), /batch/);
  const badFeatures = tvmjsManifestPayload();
  badFeatures.requiredFeatures = 'webgpu';
  assertValiError(() => v.parse(TvmjsManifestSchema, badFeatures), /requiredFeatures/);
});

// --- WholeOnnxRuntimeManifest (shape from public/runtimes/lc0-tvmjs-webgpu/.../v3-detached/manifest.json) ---

function wholeOnnxManifestPayload() {
  return {
    schema: 'lc0_browser.lc0_tvmjs_webgpu_bundle.v1',
    modelFamily: 't1-256x10-distilled-swa-2432500',
    dtype: 'f16',
    target: 'webgpu',
    requiredFeatures: ['webgpu', 'shader-f16'],
    runtime: { tvmjsBundle: 'tvmjs.bundle.js', tvmjsRuntimeWasm: 'tvmjs_runtime.wasm', note: 'extra runtime note' },
    parameterStrategy: {
      current: 'detached-tensor-cache',
      note: 'Weights live only in the tensor-cache sidecar.',
      tensorCacheApis: { browserFetchTensorCache: 'tvm.fetchTensorCache' },
    },
    tensorCache: {
      manifest: 'tensor-cache/tensor-cache.json',
      directory: 'tensor-cache',
      shardCount: 2,
      totalBytes: 40216834,
      files: ['tensor-cache/params_shard_0.bin', 'tensor-cache/params_shard_1.bin'],
    },
    compilerProvenance: { gitDirty: true, emscripten: { emcc: 'emcc' } },
    models: [{ batch: 8, wasm: 't1.batch8.tvmjs.wasm', probe: 't1.batch8.json', bytes: 1000, sha256: 'c'.repeat(64) }],
    files: [{ path: 'tvmjs.bundle.js', bytes: 2000, sha256: 'd'.repeat(64) }],
  };
}

test('whole-onnx runtime manifest parses a realistic payload with extra fields', () => {
  const payload = wholeOnnxManifestPayload();
  assert.deepEqual(v.parse(WholeOnnxRuntimeManifestSchema, payload), payload);
});

test('whole-onnx runtime manifest rejects missing models and malformed entries', () => {
  const noModels = wholeOnnxManifestPayload();
  delete noModels.models;
  assertValiError(() => v.parse(WholeOnnxRuntimeManifestSchema, noModels), /models/);
  const noWasm = wholeOnnxManifestPayload();
  noWasm.models = [{ batch: 8 }];
  assertValiError(() => v.parse(WholeOnnxRuntimeManifestSchema, noWasm), /wasm/);
  const badRuntime = wholeOnnxManifestPayload();
  badRuntime.runtime = { tvmjsBundle: 42 };
  assertValiError(() => v.parse(WholeOnnxRuntimeManifestSchema, badRuntime), /tvmjsBundle/);
  const badFile = wholeOnnxManifestPayload();
  badFile.files = [{ sha256: 'd'.repeat(64) }];
  assertValiError(() => v.parse(WholeOnnxRuntimeManifestSchema, badFile), /path/);
});

// --- SquareFormerMeta (shape from public/models/bt4_anneal_muon_best.meta.json) ---

function squareFormerMetaPayload() {
  return {
    kind: 'squareformer_v2',
    input_dim: 112,
    token_features: 24,
    compact_feature_schema: 'compact_square_tokens_lc0_repetition_v1',
    input_mode: 'embedding',
    input_format: 'compact_uint8_tokens',
    policy_size: 20480,
    layers: 6,
    d_model: 128,
    history_plies: 7,
    relation_bias: true,
    av_head_exported: false,
    action_value_move_encoding: null,
    max_legal_moves: 128,
    onnx_fixed_legal_moves: 128,
    outputs: ['policy', 'wdl', 'q', 'hidden'],
    onnx_dynamic_batch: true,
    board_normalization: 'stm_white_rankflip_v1',
    input_index_dtype: 'int64',
    attack_summary_feature_count: 28,
    attack_summary_schema: 'threatgraph_square_summary_v1',
    attack_summary_scale: 8.0,
    training_recipe: { optimizer: 'muon_adamw', lr: 5e-6 },
  };
}

test('squareformer meta parses a realistic payload and normalizes null encoding', () => {
  const payload = squareFormerMetaPayload();
  const parsed = v.parse(SquareFormerMetaSchema, payload);
  assert.equal(parsed.kind, 'squareformer_v2');
  assert.equal(parsed.action_value_move_encoding, undefined, 'null action_value_move_encoding normalizes to undefined');
  assert.equal(parsed.attack_summary_schema, 'threatgraph_square_summary_v1');
  assert.deepEqual(parsed.training_recipe, payload.training_recipe, 'loose schema preserves unknown fields');
});

test('squareformer meta rejects missing required fields and wrong primitive types', () => {
  const noInputDim = squareFormerMetaPayload();
  delete noInputDim.input_dim;
  assertValiError(() => v.parse(SquareFormerMetaSchema, noInputDim), /input_dim/);
  const badKind = squareFormerMetaPayload();
  badKind.kind = 'transformer';
  assertValiError(() => v.parse(SquareFormerMetaSchema, badKind), /kind/);
  const badHistory = squareFormerMetaPayload();
  badHistory.history_plies = '7';
  assertValiError(() => v.parse(SquareFormerMetaSchema, badHistory), /history_plies/);
  const badOutputs = squareFormerMetaPayload();
  badOutputs.outputs = 'policy';
  assertValiError(() => v.parse(SquareFormerMetaSchema, badOutputs), /outputs/);
});

// --- policyOnlyBrowser representative input fixtures (fixtures/lc0/*.json) ---

test('fen input fixtures parse realistic entries with extra fields', () => {
  const payload = [
    {
      id: 'startpos',
      description: 'Standard start position for first root-prior smoke checks.',
      fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
      expectedLegalMoves: ['d2d4', 'g1f3', 'e2e4'],
    },
  ];
  assert.deepEqual(v.parse(FenInputFixtureListSchema, payload), payload);
});

test('fen input fixtures reject entries missing fen or with wrong types', () => {
  assertValiError(() => v.parse(FenInputFixtureListSchema, [{ id: 'x' }]), /fen/);
  assertValiError(() => v.parse(FenInputFixtureListSchema, [{ id: 7, fen: 'x' }]), /id/);
});

test('history input fixtures parse realistic entries with extra fields', () => {
  const payload = [
    {
      id: 'italian-four-ply-history',
      description: 'Non-repeating opening history.',
      startFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
      moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6'],
      finalFen: 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
    },
  ];
  assert.deepEqual(v.parse(HistoryInputFixtureListSchema, payload), payload);
});

test('history input fixtures reject entries missing moves or with wrong types', () => {
  assertValiError(() => v.parse(HistoryInputFixtureListSchema, [{ id: 'x' }]), /moves/);
  assertValiError(() => v.parse(HistoryInputFixtureListSchema, [{ id: 'x', moves: ['e2e4', 42] }]), /moves/);
  assertValiError(() => v.parse(HistoryInputFixtureListSchema, [{ id: 'x', moves: [], startFen: 9 }]), /startFen/);
});
