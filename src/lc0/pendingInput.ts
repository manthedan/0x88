/** The raw transfer is its identity, so acknowledging an older input cannot retire a newer one. */
export interface InputTransfer {
  key: string;
  id: string;
}
export interface PendingInput {
  urlKeys: Array<'fen' | 'pgn'>;
  transfers: InputTransfer[];
}
export type PendingInputEvent = { type: 'applied'; input: PendingInput } | { type: 'saved' | 'cleared' | 'discarded' | 'save-failed' | 'clear-failed' };

/** Pure lifecycle: controllers perform the returned acknowledgement effects exactly once. */
export function transitionPendingInput(
  pending: PendingInput | null,
  event: PendingInputEvent,
): { pending: PendingInput | null; acknowledged: PendingInput | null } {
  if (event.type === 'applied') {
    return {
      pending: {
        urlKeys: [...new Set([...(pending?.urlKeys ?? []), ...event.input.urlKeys])],
        transfers: [...(pending?.transfers ?? []), ...event.input.transfers].filter(
          (transfer, index, all) => all.findIndex((other) => other.key === transfer.key && other.id === transfer.id) === index,
        ),
      },
      acknowledged: null,
    };
  }
  if (event.type === 'save-failed' || event.type === 'clear-failed') return { pending, acknowledged: null };
  return { pending: null, acknowledged: pending };
}
