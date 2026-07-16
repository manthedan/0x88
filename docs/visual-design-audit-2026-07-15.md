# 0x88 visual design audit

**Date:** 2026-07-15  
**Reviewed commit:** `d49a52c`  
**Branch:** `audit/visual-design`  
**Worktree:** `/Users/macthedan/projects/lc0_browser/leelaweb-visual-design-audit`  
**Surfaces:** Home, Play, Analysis, Arena, Docs; light and dark themes; 1280 px, 390 px, and 320 px widths

## Executive verdict

0x88 already has a credible visual identity: warm chess-book neutrals, olive accents, editorial serif headings, compact monospace telemetry, and a board-first product layout. The application and documentation feel more deliberate than a typical generated interface.

The main exception is the landing page. Its oversized gradient hero, highlighted phrase, two-button CTA, and three rounded feature cards with emoji tiles reproduce a common AI/SaaS template. The visual language becomes much more specific and convincing as soon as the user enters the chess tools.

The highest-value work is not a wholesale redesign. Preserve the warm palette and serif/mono/sans system, then:

1. Make the landing page unmistakably about the 0x88 chess workbench rather than a generic software product.
2. Fix desktop control discoverability, narrow-screen header overflow, and low-contrast microcopy.
3. Reduce nested cards, decorative rounding, and pill treatments where rules and typography would communicate hierarchy more cleanly.

## Scorecard

Scores use a 1–4 scale: 1 = poor, 2 = uneven, 3 = good, 4 = exceptional.

| Pillar | Score | Assessment |
|---|---:|---|
| Copywriting | 3/4 | Concrete, technically trustworthy, and unusually strong in Docs. Repetition and generic landing-page framing weaken the top-level story. |
| Visuals | 3/4 | Strong chess boards and cohesive product surfaces. Landing imagery and icons are generic and inconsistent. |
| Color | 3/4 | Distinctive warm/olive palette with a convincing dark mode. Several muted tokens fail normal-text contrast and some light-only borders leak into dark mode. |
| Typography | 3/4 | Serif, sans, and mono roles suit the product. Tiny uppercase labels are overused and too faint. |
| Spacing | 3/4 | Consistent panel rhythm and good mobile stacking. Desktop sidebars and empty states create avoidable clipping or blank space. |
| Experience design | 2/4 | Core tools are understandable, but essential desktop actions can begin below the visible nested scroller; the 320 px header overflows. |
| **Overall** | **17/24** | **Good foundation with several high-leverage corrections.** |

## What should be preserved

- **The warm parchment and olive palette.** It feels closer to a chess publication or physical study desk than a generic blue/purple software dashboard.
- **The board-first hierarchy.** Play, Analysis, and Arena immediately show the object the user came to manipulate.
- **The editorial/technical type pairing.** Serif titles add authority; monospace works well for FEN, PGN, engine output, and labels.
- **The restrained shadow use inside the application.** Most hierarchy comes from contrast and rules, not floating glass cards.
- **The Docs page voice and composition.** It is the strongest brand surface: direct writing, measured line length, clear sections, and meaningful technical detail.
- **Light/dark parity as a product feature.** Both themes feel intentional rather than mechanically inverted.

## Priority findings

### 1. Essential desktop actions are hidden inside a nested sidebar scroller

**Severity:** High  
**Pillars:** Experience design, spacing

The desktop application sidebars use `position: sticky`, a viewport-derived `max-height`, and `overflow-y: auto`:

- `src/routes/app/play/+page.svelte:110-112`
- `src/routes/app/analysis/+page.svelte:237-239`
- `src/routes/app/arena/+page.svelte:214-216`

At a 1280 × 580 viewport, Arena's sidebar is 1,111 px tall inside a 494 px scrollport. The **Start match** action begins around y=751 and is not visible. The scrollbar can also be visually hidden by the operating system, so the panel does not clearly advertise that it scrolls.

This is the largest practical usability problem in the audit. Analysis happens to place **Analyze** first, but Arena and portions of Play bury primary actions.

**Recommendation**

- Keep the primary action row visible: make it sticky at the top or bottom of the sidebar, outside the scrolling settings body.
- Prefer a single page scroll when practical. If the sidebar must scroll independently, add a visible edge/fade affordance and avoid placing the only start action below the fold.
- On shorter desktop viewports, reduce vertical settings density or collapse secondary sections by default.

**Acceptance check**

At 1280 × 580 and 1024 × 768, the current primary action for every app mode is visible without scrolling an unmarked nested region.

### 2. The shared header overflows at 320 px

**Severity:** High  
**Pillars:** Experience design, spacing

At 320 px, the document width is 353 px. The theme button ends near x=329, so it is clipped and the page gains horizontal overflow. The issue comes from five persistent nav links plus the brand and theme control (`public/app-shell.css:344-355`).

**Recommendation**

Use a real narrow-header mode below approximately 360 px:

- Keep the brand mark, current section, and theme control visible.
- Move non-current destinations into a compact menu, or hide `Home` behind the brand and abbreviate one secondary label.
- Do not solve this only by shrinking type below 12 px or reducing touch targets.

**Acceptance check**

`document.documentElement.scrollWidth === window.innerWidth` at 320 px, with a minimum 44 px effective target for menu/theme controls.

### 3. Muted text is too faint, especially at 9–12 px

**Severity:** High  
**Pillars:** Color, typography

The light theme uses `--muted: #8a7d66` and `--muted-2: #9b8e76` (`public/app-shell.css:20`). Approximate contrast ratios are:

- `#8a7d66` on `#f3eee4`: **3.49:1**
- `#9b8e76` on `#fcf9f2`: **3.06:1**

Many labels using these colors are only 9–12 px. The dark `--muted-2` is also about **4.06:1** on a dark panel. These combinations miss 4.5:1 for normal text and create the washed-out, low-information look often associated with over-styled generated UIs.

**Recommendation**

- Raise small-label contrast to at least 4.5:1 in both themes.
- Use `--muted-2` for truly nonessential ornament only, not form labels, table headers, or section names.
- Increase critical uppercase labels from 9–10 px to at least 11–12 px and reduce letter spacing slightly.

### 4. The landing page carries most of the “AI design” smell

**Severity:** Medium  
**Pillars:** Visuals, copywriting

The landing page combines several now-generic patterns:

- layered radial gradients and a decorative grid (`src/routes/+page.svelte:74-89`)
- a huge serif claim with one olive-highlighted phrase
- a primary/secondary CTA pair
- three equal rounded cards (`src/routes/+page.svelte:127-146`)
- emoji inside rounded tinted squares (`src/routes/+page.svelte:40,45,50,138-143`)
- generic framing such as “What's inside” and “Three modes, one browser tab.”

No one element is inherently wrong, but the combination looks generated because it could introduce almost any developer tool after swapping the nouns.

**Recommendation: make the hero a chess workbench**

- Replace ambient gradient decoration with a real product composition: a cropped board, a compact engine line, and one comparison datum or arena pairing.
- Turn the faint grid into an actual 0x88 indexing motif, not a generic background texture.
- Replace emoji with a consistent one-color icon set or miniature board/engine artifacts. The magnifying-glass emoji is particularly platform-dependent.
- Use flatter, divider-led mode entries rather than three identical floating cards.
- Make the copy concrete: describe the engines users can run and the decision each mode helps them make.

### 5. Hard-coded light borders leak into dark mode

**Severity:** Medium  
**Pillars:** Color, visual consistency

Several components use `#e6decc` directly instead of a token:

- `public/app-shell.css:246,287`
- `src/routes/app/play/+page.svelte:141`
- `src/routes/app/analysis/+page.svelte:401`
- `src/routes/app/arena/+page.svelte:261,342`

The Play move list has a conspicuously bright outline in dark mode. These literals also make future theme tuning harder.

**Recommendation**

Replace light-only literals with `var(--rule)`, `var(--rule-strong)`, or a dedicated semantic border token. Audit direct `white`, `#fff`, and light-only surface values at the same time.

### 6. Mobile Docs delays the article behind a long table of contents

**Severity:** Medium  
**Pillars:** Experience design, spacing

At widths below 860 px, the full table of contents becomes a static block before the article (`src/routes/docs/+page.svelte:776-782`). On a 390 px viewport, users must pass a long navigation list before reaching “The pages.”

**Recommendation**

Convert the mobile TOC to a collapsed `<details>` or a compact “On this page” selector. Keep the desktop sticky rail unchanged.

### 7. Browser capability status consumes disproportionate mobile space

**Severity:** Medium  
**Pillars:** Spacing, experience design

The capability strip is useful, but at 390 px its status pills wrap into a two-line, roughly 68 px block before the board. It appears on every application page (`src/lib/components/BrowserCapabilities.svelte:49-57`).

**Recommendation**

On mobile, show one compact summary such as “WebGPU + threads ready” with details collapsed. Promote a warning only when a missing capability changes expected behavior.

### 8. Arena's empty result region creates a large unexplained gap

**Severity:** Low  
**Pillars:** Spacing, experience design

The game log reserves a fixed 160 px even when empty (`src/routes/app/arena/+page.svelte:327`). On mobile this reads as a broken or unfinished panel between “No games played yet” and “Export PGN.”

**Recommendation**

Use a small purposeful empty state, then expand the log when games exist. Avoid reserving scroll-region height without a visible frame or explanation.

### 9. Too many rounded containers flatten hierarchy

**Severity:** Low  
**Pillars:** Visuals, spacing

Panels, status strips, capability boxes, details controls, buttons, badges, inputs, and mode cards all use rounded bordered containers. This is consistent, but consistency alone does not create hierarchy. It also contributes to the generated-dashboard impression.

**Recommendation**

- Reserve 10–14 px radii for major panels and primary actions.
- Use 4–8 px radii for fields and compact controls.
- Replace some nested boxes with whitespace, section rules, or a background shift.
- Avoid pills except for statuses, compact filters, or genuinely binary state.

## AI-smell inventory

| Pattern | Current risk | Preferred treatment |
|---|---|---|
| Giant gradient hero | Medium/high | Product screenshot or live workbench fragment with a subtle 0x88 board-index motif |
| Accent-colored phrase in headline | Medium | One strong concrete headline; use color on data or action rather than an arbitrary phrase |
| Three equal feature cards | High | Editorial mode index with varied, task-specific previews |
| Emoji icon tiles | High | Consistent custom SVGs, piece silhouettes, or engine logos |
| Excessive pills/rounded rectangles | Medium | Fewer containers, more rules and typographic grouping |
| Tiny uppercase micro-labels | Medium | Larger, higher-contrast labels; reserve uppercase mono for telemetry |
| Generic “private/local/no server” repetition | Medium | State it once prominently, then use space for mode-specific benefits |
| Purple/blue neon AI palette | None | Keep the existing warm/olive palette |
| Glassmorphism | None | Do not introduce it |
| Gratuitous animated gradients/glows | None | Keep motion functional and minimal |

## Recommended design direction

### “Editorial chess workbench”

The product should feel like a serious chess publication merged with a compact engine console—not an AI startup landing page and not a dense enterprise dashboard.

**Visual principles**

1. **The board is the hero.** Show a real position, real line, or real engine comparison wherever visual storytelling is needed.
2. **Data earns color.** Olive indicates active state, best line, or primary action; it should not decorate every surface.
3. **Typography before containers.** Use section headings, rules, and alignment before adding another card.
4. **Monospace means machine output.** Reserve it for FEN, PGN, evaluation, engine names where appropriate, and compact metadata.
5. **Chess-specific iconography only.** Piece silhouettes, board coordinates, eval bars, and engine marks will age better than emoji.
6. **Quiet motion.** Board moves, analysis progress, and state changes may animate; cards and backgrounds do not need theatrical motion.

## Suggested implementation sequence

### Wave 1 — usability and theme integrity

1. Fix the 320 px header overflow.
2. Keep Play/Analysis/Arena primary actions visible on short desktop viewports.
3. Raise muted-text contrast and minimum micro-label size.
4. Replace hard-coded light borders with semantic tokens.
5. Collapse the mobile capability strip and Docs TOC.
6. Remove Arena's empty 160 px log gap.

### Wave 2 — remove landing-page AI smells

1. Build one product-specific hero composition from a board, engine line, and comparison/status element.
2. Replace the three emoji cards with an editorial mode index and real UI previews.
3. Tighten landing copy and remove repeated privacy/local-runtime claims.
4. Make the 0x88 mark and grid motif communicate the chess-board indexing idea.

### Wave 3 — systematic polish

1. Define a two-level radius system and simplify nested panels.
2. Standardize compact icon buttons and replace text-symbol navigation where clarity suffers.
3. Add visual-regression screenshots for 1280 × 800, 1024 × 768, 390 × 844, and 320 × 700 in both themes.
4. Add contrast checks for all design tokens used below 18 px.

## Remediation status

Implemented on `audit/visual-design` after the audit:

- Replaced the generic gradient/card/emoji landing page with an editorial chess workbench, a real board position, and engine comparison output.
- Removed the desktop nested sidebar scrollers and moved Arena's primary action into the initially visible matchup section.
- Added a dedicated 320 px navigation menu; verified zero horizontal document overflow.
- Raised light and dark muted-text contrast and increased the usefulness of semantic color tokens.
- Replaced hard-coded light borders with theme-aware rule tokens.
- Collapsed browser capability status to one mobile badge and converted the mobile Docs TOC to a compact disclosure.
- Removed Arena's reserved empty log gap and simplified footer copy/navigation.
- Reduced global radii and shifted landing-page hierarchy from cards to typography and section rules.

## Verification performed

- Production static build: `npm run build:client` — passed.
- Rendered-browser review of Home, Play, Analysis, Arena, and Docs.
- Light theme at 1280 px and 390 px.
- Dark-theme spot checks on Home, Play, and Analysis.
- Narrow-width check at 320 px.
- Runtime-loaded states observed for boards, engine selectors, analysis lines, and capability badges.
- Full project test suite: 589 passed, 0 failed, 3 skipped.
- Final browser diagnostics: no console errors, no page errors, and no failed same-origin requests.
- Audit screenshots are retained in `.local-dev-artifacts/visual-design-audit-2026-07-15/`.
- Remediation screenshots are retained in `.local-dev-artifacts/visual-design-remediation/`.

## Top three fixes

1. Keep primary app actions visible and eliminate the unmarked nested-scroll trap.
2. Replace the landing page's generic gradient/card/emoji composition with a real 0x88 chess workbench visual.
3. Fix narrow-header overflow and low-contrast microcopy across both themes.
