# Ledger — Design System

The design reference for the Agentic Data Analysis console, the plate, and the
shareable HTML report. Change a token here and in the code together.

---

## The brief

**What this is.** An autonomous statistical analyst. You give it a dataset and a
question in plain English; it plans the analysis, runs the tests, trains the
models, and — the part that matters — tells you where it has fooled itself.

**Who reads it.** Anyone with a spreadsheet and a question — not a
data-science audience. A shopkeeper checking sales, a teacher checking test
scores, a coach checking game stats, a student checking a survey. They don't
know what "generalisation" or "train-test gap" means, and they shouldn't
have to. They care whether the answer is trustworthy and what to do about it.

**The concept.** The console is a warm, familiar ledger — the kind of book
anyone keeps to track something that matters to them, redrawn as software.
Nothing on the page should feel like it belongs to an engineer: rounded
cards instead of ruled plates, a friendly rounded typeface instead of a
technical grotesque, soft warm shadows instead of hard offset ink-stamps.
The seven-step pipeline still plots itself as the run progresses, but in
plain language ("Reading Your File," not "Dataset Ingestion") and warm
lamp-lit color instead of drafting blue.

**Explicitly rejected.** The previous "Drafting Table" identity — mineral
stock and architectural blueprint, IBM Plex Mono numerals, square corners on
everything, hard offset shadows, a 28px quadrille grid across the whole page.
That system was built for analysts defending a number to another analyst. It
is legible, precise, and completely wrong for someone opening this app
between customers. None of it survives here, except the discipline: every
choice below is deliberate, not a framework default.

---

## Tokens — ink

Seven colours, in two modes (Day and Night). Two of them are pens; the rest
are the paper the pens write on.

| Token         | Day       | Night     | What it is                                    |
| ------------- | --------- | --------- | ---------------------------------------------- |
| `--stock`     | `#f7eedd` | `#241c14` | The page itself — warm cream / warm dark brown. |
| `--sheet`     | `#fffbf2` | `#2f251a` | The lifted card: panels, tiles, tables.        |
| `--ink`       | `#3a2b1e` | `#f3e9d8` | Body text, headings, linework.                 |
| `--graphite`  | `#8a7660` | `#b8a688` | Secondary type, captions.                      |
| `--pen`       | `#a34f20` | `#f0a24a` | The action pen. Buttons, links, what's active. |
| `--risk`      | `#a33526` | `#e2685a` | The risk pen. A number that may not hold.      |
| `--positive`  | `#5b8c5a` | `#7fb77e` | Good news — a metric that came back healthy.   |

Supporting hairlines: `--rule` (`#e4d4bc` / `#4a3c28`), `--rule-faint`
(`#eee3cb` / `#3a2e1f`), `--accent` (`#e08a3e` / `#d99a4e`, decorative only).

**The rule about red.** `--risk` means exactly one thing: this number may not
hold. It is never used for emphasis or to make something look important.

**Contrast is verified, not eyeballed.** Every button-fill/text pairing in
this system was checked against WCAG AA (4.5:1 for body text, 3:1 for large
or bold text) before being chosen — see the `--pen` values above, which are
deliberately darker (Day) or paired with dark text (Night) so buttons stay
readable, not just pretty.

Both pens are flat. There are no gradients anywhere in this system.

Chart categories extend the two pens with four warm plot inks:
Day `#a34f20 #a33526 #c08a2e #5b8c5a #8a7660 #b5714a`.

---

## Tokens — type

Two families, both rounded and friendly — nothing expanded, nothing
condensed, nothing that reads as a technical stencil.

**Baloo 2** (weights 500–800) — headings, the hero, section titles, card
titles. Warm and a little playful; it's what makes the page feel handmade
rather than issued.

**Mukta** (weights 400–700) — everything else: body copy, labels, buttons,
table cells. Built for Latin+Devanagari pairing, so if this product ever adds
Hindi or another regional script the type system doesn't have to change.

Mono type (`Cascadia Code`/`Consolas`) is reserved for literal `<code>`/`<pre>`
blocks only — it is never reached for to make a number or a label look
technical. That was the old system's habit; it is exactly the habit a
product built for a general, non-expert audience should drop.

### Scale

| Role          | Size                       | Weight |
| ------------- | -------------------------- | ------ |
| Display       | `clamp(36px, 5.6vw, 64px)` | 800    |
| Section (h2)  | 27px                       | 700    |
| Sub-head (h3) | 19px                       | 700    |
| Minor (h4)    | 15.5px                     | 700    |
| Body          | 15.5px, lh 1.65, max 72ch  | 400    |
| Stat tile     | 26px                       | 700    |

Everything is left-aligned and ragged right. Prose never exceeds 72–74ch.

---

## Structure, not decoration

- **Quick Facts bar** — a rounded card under the hero, cells divided by soft
  hairlines: run state, mode, dataset, model. Same information the old
  "datum line" carried, now a card instead of a ruled bar.
- **Section head** — the title sits on a 2px warm rule, sentence case.
- **Stat tile** — a rounded, softly shadowed card. Not a ruled gauge strip.
  A flagged stat gets a risk-tinted border, not a color swap on a rule.
- **Steps list** — numbered rows, each number in a small round chip. The
  pipeline genuinely is a sequence of seven steps; the chip fills in the pen
  color as each one completes.
- **Callout cards** — insights, warnings, and recommendations hang off a
  colored left rule on a tinted rounded card, like a sticky note rather than
  a reviewer's margin mark. Graphite for a finding, pen for an action, risk
  for a warning.

### Shape and elevation

`border-radius: 14px` on cards, panels, and expanders; full pill radius on
buttons and tabs. Panels are lifted with a soft, blurred, warm-toned shadow —
the opposite of the old hard offset stamp. Paper does glow, a little, here.

The page ground is a flat warm fill. No repeating grid texture — that
quadrille was the single clearest "engineering tool" signal in the old
system and it is gone entirely.

---

## Motion

Unchanged in spirit from the previous system: one orchestrated moment on
load (the plate's entrance, still built in `ui/assets/pipeline_3d.js` with
GSAP), and nothing else uncommanded. The hero headline now rises and fades
in rather than performing a pen-strike clip-path reveal — softer, and still
gated on `prefers-reduced-motion: reduce`.

The render loop is gated on an `IntersectionObserver`, `document.hidden`,
and a dirty flag — an idle plate costs no frames.

---

## The plate

`ui/pipeline_3d.py` owns the state contract; `ui/assets/pipeline_3d.js` draws
it. The 3D view keeps its abstract stage geometry (a sheet, a crystal, a box,
a torus, decomposing cubes, stacked sheets) — rebuilding that into literal,
everyday iconography is future work, not part of this pass. What changed
here is everything the 3D view *shares* with the rest of the console: warm
lamp-like lighting instead of cool drafting light, the Day/Night palette
instead of Mineral/Blueprint, a pill-shaped view toolbar instead of square
CAD buttons, and plain-language stage labels fed in from `app.py`.

Because the geometry itself is still abstract, **the plain-language steps
list below the plate is the primary, legible account of progress** — the 3D
view is ambient flavor on top of it, not the only place the run state lives.
This is a deliberate scope boundary: a full redesign of the 3D metaphor into
something more literal (a folder, a magnifying glass, a document — visuals
that map onto anyone's everyday world rather than an abstract technical
drawing) is a larger project than this reskin and should be scoped
separately if wanted.

**Degradation.** No WebGL context means a plain-text message pointing at the
steps list, which carries the same information.

**Access.** The canvas is focusable, labelled `role="img"` with a plain-
language summary of the run, and orbits on the arrow keys; `[` and `]` step
through the stages and `Home` recentres.

---

## Writing

Plain verbs, sentence case, no filler. Say what happened and what to do —
and say it the way you'd explain it to someone who has a question about
their own data, not the way you'd write it in a paper.

- Headline: "We check every answer twice." — trustworthy, not clever.
- The seven steps: Reading Your File → Understanding Your Question → Running
  the Numbers → Making Sense of It → Double-Checking → Solving the Tricky
  Parts → Writing Your Report. These names are the single source of truth in
  `STAGE_DEFS` (`app.py`) and flow through to the plate and the steps list —
  change them in one place.
- Errors state the failure and the fix, and do not apologise.
- Empty states are an invitation: "Let's see what your data shows."
- Buttons name the action and keep that name through the flow.
- The eight helper agents get a plain job-title name ("Planner," "File
  Checker," "Fact-Checker," "Model Builder," "Reality-Checker,"
  "Double-Checker," "Detail Handler," "Report Writer") instead of an
  engineering title. Their "mission/rule/uses/found" fields in
  `_render_agent_deep_dive` (`app.py`) explain what they do in plain
  language — no named papers, no bare statistical terms without a plain
  gloss next to them.

**Extent of the plain-language pass.** The "Your Helpers" tab and the "Full
Details" tab (formerly "Statistical & ML Lab") are both rewritten into plain
language now — headers, captions, and agent descriptions. The one place
technical terms remain on purpose is inside genuine data tables in "Full
Details" (precision/recall/F1, p-values) — renaming a standard statistical
term in a data table loses meaning without adding real explanation, so
those keep their accepted names but get a plain-language caption next to
them instead (see "Accuracy by Category" and "Is the Pattern Real?" in
`app.py`). That's a deliberate line, not a leftover.

---

## Don't

- Reach for mono to make a label look technical.
- Use a hard, unblurred offset shadow.
- Use `--risk` for anything other than a result that may not hold.
- Introduce a third pen, or a gradient.
- Ship a color pairing without checking contrast — verify, don't eyeball.
- Bring back the quadrille grid, square corners, or IBM Plex Mono. That
  system is retired.
