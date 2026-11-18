// Compile-time checks for the per-route el() typing; svelte-check runs this file, nothing imports it.
import { el as analysisEl } from './analysisDom.ts';
import { el as playEl, selectEl as playSelectEl } from './playDom.ts';

export function typedLookupChecks(dynamicId: string): void {
  playSelectEl('engineSelect');
  analysisEl('pvPreview');
  playEl(dynamicId);
  for (const id of ['navFirst', 'navLast'] as const) playEl(id);
  // @ts-expect-error -- pvPreview is owned on Analysis, not Play.
  playEl('pvPreview');
  // @ts-expect-error -- a typo is not an owned id.
  playSelectEl('engineSelekt');
}
