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
