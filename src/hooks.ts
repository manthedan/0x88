import type { Reroute } from '@sveltejs/kit';
import { refuseLabRoute } from './lib/buildScope.ts';

// __BUILD_SCOPE__ is defined from BUILD_SCOPE in vite.config.ts; outside product builds this folds away.
export const reroute: Reroute | undefined = __BUILD_SCOPE__ === 'product' ? refuseLabRoute : undefined;
