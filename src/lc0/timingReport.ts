export function roundedNumericRecord(value: unknown, digits = 4): Record<string, number> | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]))
    .map(([key, numberValue]) => [key, Number(numberValue.toFixed(digits))] as const);
  return entries.length ? Object.fromEntries(entries) : undefined;
}

/** Keep missing kernel attribution explicit without admitting nonnumeric values to timing statistics. */
export function roundedBackendTimingRecord(value: unknown, digits = 4): Record<string, number | string | null> | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const record = value as Record<string, unknown>;
  const result: Record<string, number | string | null> = { ...roundedNumericRecord(value, digits) };
  for (const key of ['ortKernelProfileStatus', 'ortKernelProfileReason']) {
    if (typeof record[key] === 'string') result[key] = record[key];
  }
  if (record.ortKernelGpuMs === null) result.ortKernelGpuMs = null;
  return Object.keys(result).length ? result : undefined;
}
