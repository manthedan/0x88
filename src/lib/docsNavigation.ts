/** Resolve the last heading above the reading line, including nested topics. */
export function activeDocsTopic(
  headings: readonly { id: string; top: number }[],
  readingLine: number,
  atPageEnd = false,
): string | undefined {
  if (atPageEnd) return headings.at(-1)?.id;
  let current = headings[0]?.id;
  for (const heading of headings) {
    if (heading.top > readingLine) break;
    current = heading.id;
  }
  return current;
}

/** A single open topic list leaves the other group headings within reach. */
export function toggleDocsGroup(expandedId: string | null, groupId: string): string | null {
  return expandedId === groupId ? null : groupId;
}

export function docsSearchShortcut(platform: string): { text: string; label: string } {
  return /Mac|iPhone|iPad|iPod/.test(platform)
    ? { text: '⌘K', label: 'Command K' }
    : { text: 'Ctrl K', label: 'Control K' };
}
