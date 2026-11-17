import { BIG_NETS } from './bt4Engine.ts';
import { isMemoryConstrainedDevice } from './deviceMemoryClass.ts';
import type { EngineRow } from './engineCatalog.ts';
import { type TournamentMode, tournamentPairings } from './tournament.ts';

type Seat = Pick<EngineRow, 'family' | 'variant'>;
export const ARENA_BIG_NET_MEMORY_REASON = 'Two large nets at once can crash this device.';

function isBigNetSeat(seat: Seat): boolean {
  return seat.family === 'lc0' && Object.hasOwn(BIG_NETS, seat.variant);
}

/** Seat indices, not net identities: two seats using the same net also count. */
export function arenaBigNetPairingBlocked(mode: TournamentMode, seats: readonly Seat[], constrained = isMemoryConstrainedDevice()): boolean {
  if (!constrained) return false;
  return tournamentPairings(
    mode,
    seats.map((_, index) => String(index)),
  ).some(({ a, b }) => isBigNetSeat(seats[Number(a)]) && isBigNetSeat(seats[Number(b)]));
}

/** Only opponents matter: a CPU-led gauntlet may contain several large nets. */
export function arenaBigNetSeatBlocked(mode: TournamentMode, seats: readonly Seat[], index: number, constrained = isMemoryConstrainedDevice()): boolean {
  if (!constrained) return false;
  return tournamentPairings(
    mode,
    seats.map((_, seat) => String(seat)),
  ).some(({ a, b }) => (Number(a) === index ? isBigNetSeat(seats[Number(b)]) : Number(b) === index && isBigNetSeat(seats[Number(a)])));
}
