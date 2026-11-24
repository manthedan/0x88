<script lang="ts">
// Per-page link-preview tags (Open Graph and Twitter cards) for iMessage, Slack,
// Discord and others. The shared image and site-wide tags live in src/app.html.
// Pages are prerendered, so these reach crawlers that don't run JavaScript.
import { SITE_ORIGIN } from '#lib/site.ts';

export let title: string;
export let description: string;
/** Site path with its trailing slash, e.g. '/app/analysis/'. */
export let path: string;

$: url = `${SITE_ORIGIN}${path}`;
</script>

<meta property="og:title" content={title} />
<meta property="og:description" content={description} />
<meta property="og:url" content={url} />
<meta name="twitter:title" content={title} />
<meta name="twitter:description" content={description} />
