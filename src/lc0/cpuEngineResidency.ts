// Which cached Play CPU engines to drop when the player switches engines.
// The id being used stays. Order is oldest first, so the survivor is the
// most recently retained inactive engine. Stormphrax's own switch-away
// dispose stays in the Play page; this helper does not special-case it.

export function inactiveCpuEnginesToEvict(residentIdsOldestFirst: readonly string[], activeId: string, cap = 1): string[] {
  const inactive = residentIdsOldestFirst.filter((id) => id !== activeId);
  const overflow = inactive.length - Math.max(0, cap);
  if (overflow <= 0) return [];
  return inactive.slice(0, overflow);
}
