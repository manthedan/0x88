import { engineBrushes } from './analysisFormat.ts';

/** Keep replay colours tied to recorded identities, never live workers or chart tabs. */
export function arenaChartGroups<T extends { engineId: string; engineName: string; ply: number }>(samples: readonly T[]) {
  const ids = [...new Set(samples.map((sample) => sample.engineId))];
  const groups = ids.flatMap((id) => {
    const own = samples.filter((sample) => sample.engineId === id);
    const parities = [...new Set(own.map((sample) => sample.ply % 2))].sort();
    const name = own[0].engineName;
    const color = engineBrushes(name).swatch;
    // Exact self-play shares a worker ID. Search parity preserves both traces,
    // including custom Black-to-move roots, without assuming parity means White.
    return ids.length === 1 && parities.length === 2
      ? parities.map((parity) => ({
          id,
          color,
          label: `${name} · ${parity === 0 ? 'first' : 'alternate'} moves`,
          samples: own.filter((sample) => sample.ply % 2 === parity),
        }))
      : [{ id, color, label: name, samples: own }];
  });
  return groups.map((group, index) => ({ ...group, dashed: groups.slice(0, index).some((other) => other.color === group.color) }));
}
