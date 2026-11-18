import { refuseInProductBuild } from '../../lib/buildScope.ts';

// VITE_BUILD_SCOPE is defined from BUILD_SCOPE in vite.config.ts.
export const load = () => refuseInProductBuild(import.meta.env.VITE_BUILD_SCOPE);
