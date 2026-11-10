export function arenaAdjudicationLabel(reason: string): string | null {
  return reason === 'max plies' ? 'Adjudicated draw · 300-ply limit' : null;
}

export function arenaAdjudicationTags(reason: string): Record<string, string> {
  return reason === 'max plies' ? { Termination: 'adjudication', Adjudication: 'Draw at the 300-ply limit' } : {};
}
