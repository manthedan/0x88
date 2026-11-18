import { refuseInProductBuild } from '../../lib/buildScope.ts';

// __BUILD_SCOPE__ is defined from BUILD_SCOPE in vite.config.ts.
export const load = () => refuseInProductBuild(__BUILD_SCOPE__);
