// Which cached Play CPU engines to drop when the player switches engines.
// The id being used stays. Order is oldest first, so the survivor is the
// most recently retained inactive engine. Stormphrax's own switch-away
// dispose stays in the Play page; this helper does not special-case it.

/**
 * Play keeps CPU engines resident between games, so their hash would carry
 * the previous game into the next one. A new game marks every resident
 * engine; the engine's next move first sends newGame (ucinewgame). The mark is
 * cleared only after the reset succeeds, so an aborted reset is retried.
 */
export class PendingCpuNewGames {
  private readonly ids = new Map<string, symbol>();

  markAll(ids: Iterable<string>): void {
    for (const id of ids) this.ids.set(id, Symbol());
  }

  forget(id: string): void {
    this.ids.delete(id);
  }

  has(id: string): boolean {
    return this.ids.has(id);
  }

  async resetIfPending(id: string, engine: { newGame?(signal?: AbortSignal): Promise<void> }, signal?: AbortSignal): Promise<void> {
    const mark = this.ids.get(id);
    if (mark === undefined) return;
    await engine.newGame?.(signal);
    // A newer game (or replacement engine with the same id) owns its own reset.
    if (this.ids.get(id) === mark) this.ids.delete(id);
  }
}

export function inactiveCpuEnginesToEvict(residentIdsOldestFirst: readonly string[], activeId: string, cap = 1): string[] {
  const inactive = residentIdsOldestFirst.filter((id) => id !== activeId);
  const overflow = inactive.length - Math.max(0, cap);
  if (overflow <= 0) return [];
  return inactive.slice(0, overflow);
}
