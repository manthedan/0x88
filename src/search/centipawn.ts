import { montyLitePuctPolicy, type SearchOptions } from './puct.ts';

// Native monty_lc0both constants; budgets and batching remain surface choices.
export const centipawnSearchOptions = {
  searchPolicy: montyLitePuctPolicy,
  cpuct: 1.5,
  cpuctSchedule: 'lc0-log',
  cpuctBase: 38739,
  cpuctFactor: 3.894,
  fpu: 0,
  fpuStrategy: 'lc0-reduction',
  fpuReduction: 0.33,
  avWeight: 0,
} satisfies SearchOptions;
