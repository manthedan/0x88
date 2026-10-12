// A worker cancel is applied immediately, while the search it names may still be
// queued behind the search that is running. Remember that id so handleSearch
// does not start it when the queue drains.

const MAX_PENDING_SEARCH_CANCELS = 32;

/** Record a cancel for a search that is not in activeSearches yet. */
export function noteSearchCancel(pending: number[], id: number): void {
  if (pending.includes(id)) return;
  pending.push(id);
  if (pending.length > MAX_PENDING_SEARCH_CANCELS) pending.shift();
}

/** True when this search was cancelled before it started. The id is forgotten. */
export function takeSearchCancel(pending: number[], id: number): boolean {
  const index = pending.indexOf(id);
  if (index < 0) return false;
  pending.splice(index, 1);
  return true;
}
