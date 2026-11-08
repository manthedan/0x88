// Browser-safe lab identity/eligibility rules. Does not import a neural runtime.
// Module lifetime survives client-side route unmounts; a pending GC-backed memory
// measurement must block ALL later gates until its real promise settles.
let pendingMemory = 0;
const memoryListeners = new Set<(pending: boolean) => void>();
export function simulMemoryPending(): boolean {
  return pendingMemory > 0;
}
export function subscribeSimulMemory(listener: (pending: boolean) => void): () => void {
  memoryListeners.add(listener);
  listener(simulMemoryPending());
  return () => {
    memoryListeners.delete(listener);
  };
}
export async function measureSimulMemory<T>(measure: () => Promise<T>): Promise<T> {
  pendingMemory++;
  for (const listener of memoryListeners) listener(true);
  try {
    return await measure();
  } finally {
    pendingMemory--;
    for (const listener of memoryListeners) listener(simulMemoryPending());
  }
}
interface EquivalenceGame {
  id: number;
  moves: string[];
  visits: number[];
  result: string;
  reason: string;
}
export interface SimulEligibilityResult {
  n: number;
  visits: number;
  maxPlies: number;
  pairs: number;
  rows: {
    control: boolean;
    physical: number;
    modelUrl: string;
    concurrency: number;
    gamesPerHour: number;
    evalsPerSecond: number;
    completed: number;
    games: EquivalenceGame[];
  }[];
}

export function simulEquivalentGames(a: EquivalenceGame[], b: EquivalenceGame[]): boolean {
  if (a.length !== 8 || b.length !== 8) return false;
  const byId = new Map(b.map((game) => [game.id, game]));
  return (
    byId.size === 8 &&
    new Set(a.map((game) => game.id)).size === 8 &&
    a.every((game) => {
      const other = byId.get(game.id);
      return (
        other &&
        game.result === other.result &&
        game.reason === other.reason &&
        JSON.stringify(game.moves) === JSON.stringify(other.moves) &&
        JSON.stringify(game.visits) === JSON.stringify(other.visits)
      );
    })
  );
}

export function simulHelps4(results: SimulEligibilityResult[], visits: number, maxPlies: number): boolean {
  const median = (values: number[]) => [...values].sort((a, b) => a - b)[1];
  return results.some((result) => {
    if (result.n !== 4 || result.visits !== visits || result.maxPlies !== maxPlies || result.pairs !== 3) return false;
    const measured = result.rows.filter((row) => !row.control && row.completed === 8);
    const serial = measured.filter((row) => row.concurrency === 1);
    const simul = measured.filter((row) => row.concurrency === 4);
    const control = result.rows.find((row) => row.control && row.completed === 8);
    if (
      serial.length !== 3 ||
      simul.length !== 3 ||
      !control ||
      control.physical !== 4 ||
      serial.some((row) => row.physical !== 1) ||
      !simul.every((row) => row.physical === control.physical && row.modelUrl === control.modelUrl && simulEquivalentGames(row.games, control.games))
    )
      return false;
    // Export drift can shorten games. Require an inference-throughput gain too, not
    // just a games/hour improvement caused by different game lengths.
    return (
      median(simul.map((row) => row.gamesPerHour)) > 1.1 * median(serial.map((row) => row.gamesPerHour)) &&
      median(simul.map((row) => row.evalsPerSecond)) > 1.1 * median(serial.map((row) => row.evalsPerSecond))
    );
  });
}

export function simulCancelGame(controllers: Map<number, AbortController>, game?: number): void {
  const target = game ?? controllers.keys().next().value;
  if (target !== undefined) controllers.get(target)?.abort();
}

export function simulGeneration(runId: string, trial: number): string {
  return `${runId}:${trial}`;
}
