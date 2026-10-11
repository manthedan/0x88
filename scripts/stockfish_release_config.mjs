// One provenance pin for the npm artifacts, reproducible builds and source archive.
export const STOCKFISH_RELEASE = {
  version: '19.0.1',
  major: '19',
  repo: 'https://github.com/nmrugg/stockfish.js.git',
  tag: 'v19.0.1',
  commit: '0d5dd7944f88c817c07a26bc4fb1905d5f9c46d6',
  nativeTag: 'sf_19',
  nativeCommit: 'edb0d9db6731067ec50ce619ff372b463bc4dd5d',
  license: 'GPL-3.0',
  toolchain: 'emscripten/emsdk:6.0.9',
  nets: [
    {
      variant: 'full',
      name: 'nn-1a298aa575a0.nnue',
      sha256: '1a298aa575a085434d29027978dc36867fe9c5bcea9376654b7a8eba1e52dfc2',
    },
    {
      variant: 'lite',
      name: 'nn-61e7af4bb97d.nnue',
      sha256: '61e7af4bb97d51eeeb25d322916f86513b5cd3a827ce189c98c6e31946f99e5b',
    },
  ],
};

export const stockfishSourcePrefix = `upstream/stockfish-js-${STOCKFISH_RELEASE.commit}/`;
export const stockfishSourceArchive = `public/stockfish/stockfish-${STOCKFISH_RELEASE.version}-corresponding-source.tar.gz`;
export const stockfishArtifactBases = ['lite-single', 'lite-single-relaxed', 'lite', 'single', ''].map(
  (suffix) => `stockfish-${STOCKFISH_RELEASE.major}${suffix ? `-${suffix}` : ''}`,
);
