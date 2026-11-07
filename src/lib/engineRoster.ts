// Landing-page engine roster. Which engines appear, their names and their
// order come from the engine registry; only the one-line roles are written
// here (sourced from the engine entries on /docs).
import { engineMonogram } from '../lc0/analysisPresentation.ts';
import { type EngineFamily, engineFamilyDefinition, engineFamilyOptions } from '../lc0/engineCatalog.ts';

export type RosterRuntime = 'gpu' | 'cpu';

export interface RosterEngine {
  id: string;
  name: string;
  monogram: string;
  role: string;
  runtime: RosterRuntime;
  docHref: string;
}

const ROLES: Partial<Record<EngineFamily, string>> = {
  lc0: 'Leela Chess Zero networks with tree search. Small, t3 and BT4 nets.',
  sf: 'Stockfish 19 alpha-beta search with NNUE, in Lite and full builds.',
  reckless: 'Rust alpha-beta engine with NNUE evaluation.',
  berserk: 'Berserk 14, a C engine with a bucketed NNUE.',
  viridithas: 'Rust NNUE engine trained without other engines’ evaluations.',
  plentychess: 'PlentyChess 8, a C++ engine with a threat-input NNUE.',
  stormphrax: 'Stormphrax 8, a C++ engine with a self-play NNUE.',
  centipawn: '0x88’s own neural engine. Experimental.',
};

/** Analysis and Arena engines, in the registry's selector order. */
export function searchEngineRoster(): RosterEngine[] {
  return engineFamilyOptions().map(({ value: family }) => {
    const definition = engineFamilyDefinition(family);
    return {
      id: family,
      name: definition.label,
      monogram: engineMonogram(definition.label),
      role: ROLES[family] ?? '',
      runtime: definition.resource.resourceClass,
      docHref: `/docs/#engines-${family === 'sf' ? 'stockfish' : family}`,
    };
  });
}

/** Human-style Play opponents that are not Analysis engines. */
export const HUMAN_OPPONENTS: RosterEngine[] = [
  {
    id: 'maia3',
    name: 'Maia3',
    monogram: engineMonogram('Maia3'),
    role: 'Plays the move a human at your chosen rating would play, from 600 to 2600.',
    runtime: 'gpu',
    docHref: '/docs/#engines-maia3',
  },
  {
    id: 'lqo',
    name: 'Leela Queen Odds',
    monogram: 'LQ',
    role: 'Starts without its queen and searches for your mistakes.',
    runtime: 'gpu',
    docHref: '/docs/#engines-lqo',
  },
];
