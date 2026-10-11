import { moveToUci } from '../chess/moveCodec.ts';
import type { BrowserUciEngine } from './browserUciEngine.ts';
import type { GameNode, GameTree } from './gameTree.ts';

export function analysisGamePosition(tree: GameTree, node: GameNode = tree.current): { fen: string; moves: string[] } {
  return {
    fen: tree.root.fen,
    moves: tree
      .lineTo(node)
      .slice(1)
      .map((entry) => moveToUci(entry.move!)),
  };
}

/** Surface-owned continuity; recreated adapters and replaced games always reset. */
export class AnalysisCpuSessions {
  private games = new WeakMap<object, GameTree>();
  private threads = new WeakMap<object, number>();

  needsReset(engine: object, game: GameTree): boolean {
    return this.games.get(engine) !== game;
  }

  prepared(engine: object, game: GameTree): void {
    this.games.set(engine, game);
  }

  invalidate(engine: object): void {
    this.games.delete(engine);
  }

  setThreads(engine: Pick<BrowserUciEngine, 'setOptions'>, threads: number): void {
    // Stockfish's Threads callback can rebuild the TT even when unchanged.
    if (this.threads.get(engine) === threads) return;
    engine.setOptions({ threads });
    this.threads.set(engine, threads);
  }
}
