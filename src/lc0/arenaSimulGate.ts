// Browser-safe lab identity/eligibility rules. Does not import a neural runtime.
export interface SimulEligibilityResult {
  n: number;
  visits: number;
  maxPlies: number;
  pairs: number;
  rows: { control: boolean; concurrency: number; gamesPerHour: number; completed: number }[];
}

export function simulHelps4(results: SimulEligibilityResult[], visits: number, maxPlies: number): boolean {
  const median = (values: number[]) => [...values].sort((a, b) => a - b)[1];
  return results.some((result) => {
    if (result.n !== 4 || result.visits !== visits || result.maxPlies !== maxPlies || result.pairs !== 3) return false;
    const measured = result.rows.filter((row) => !row.control && row.completed === 8);
    const serial = measured.filter((row) => row.concurrency === 1);
    const simul = measured.filter((row) => row.concurrency === 4);
    return serial.length === 3 && simul.length === 3 && median(simul.map((row) => row.gamesPerHour)) > 1.1 * median(serial.map((row) => row.gamesPerHour));
  });
}

export function simulGeneration(runId: string, trial: number): string {
  return `${runId}:${trial}`;
}
