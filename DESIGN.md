# Drafting Table — Design System

The design reference for the Agentic Data Analysis console, the plate, and the
shareable HTML report. Change a token here and in the code together.

---

## The brief

**What this is.** An autonomous statistical analyst. You give it a dataset and a
question in plain English; it plans the analysis, runs the tests, trains the
models, and — the part that matters — tells you where it has fooled itself.

**Who reads it.** Analysts and engineers who have to defend a number to someone
else. They care less about how clever the agent is than about whether the result
holds outside the training split.

**The concept.** The console is a draughtsman's sheet. The agent works in ink on
paper. Its reasoning is drawn, not lit: the seven-stage pipeline is an
axonometric technical drawing that plots itself as the run progresses, and the
results are ruled plates, gauge strips and specimen tables rather than cards.

**Explicitly rejected.** The previous design was a borrowed identity — obsidian
canvas, one neon-green accent, Inter and Inter Tight, tracked-out ALL-CAPS
eyebrows above every heading, pill buttons, 20px radius on everything, soft grey
shadows. Every one of those is a default rather than a choice, and none of them
had anything to do with statistics. None of them survive here.

---

## Tokens — ink

Six colours. Two of them are pens; the rest are the stock the pens draw on.

| Token        | Hex       | What it is                                          |
| ------------ | --------- | --------------------------------------------------- |
| `--stock`    | `#dcdbd3` | Mineral drafting stock. The page itself.            |
| `--sheet`    | `#efeee8` | The lifted sheet: plates, tables, panels.           |
| `--ink`      | `#171c1f` | Drawing ink. Type, rules, linework.                 |
| `--graphite` | `#54585b` | Soft pencil. Secondary type, unreached structure.   |
| `--pen`      | `#12467e` | The measurement pen. What the agent measured.       |
| `--risk`     | `#b5271a` | The risk pen. Overfit, failure, warning. Nothing else. |

Supporting hairlines: `--rule #b6b4a9`, `--rule-faint #c8c6bc`,
`--quadrille rgba(23,28,31,.045)`.

**The rule about red.** `--risk` means exactly one thing: this measurement may
not hold. A train–test gap at or above 10 points, a failed stage, a data-quality
warning. It is never used for emphasis, never for a heading, never to make
something look important.

Both pens are flat. There are no gradients anywhere in this system.

Chart categories extend the two pens with four muted plot inks:
`#12467e`, `#b5271a`, `#7a8b99`, `#c08a2e`, `#3f6f5b`, `#8e6e9e`.

---

## Tokens — type

Two families.

**Archivo** (variable, `wdth` 75–125, `wght` 400–800) — display and UI. The width
axis does the work: display sits at `wdth 118` and weight 800, body at `wdth 100`
and weight 400. An expanded grotesque set very large and very tight is the
stencil on a technical drawing.

**IBM Plex Mono** — every measured number, table figure, code block, stage index
and log line. This is functional, not decorative: statistics need tabular
alignment. Mono never carries a decorative label.

### Scale

Large jumps, few steps.

| Role          | Size                       | Settings                              |
| ------------- | -------------------------- | ------------------------------------- |
| Display       | `clamp(40px, 6.6vw, 78px)` | `wdth 118`, 800, `-.038em`, lh `.93`  |
| Section (h2)  | 30px                       | `wdth 112`, 700, `-.025em`            |
| Sub-head (h3) | 19px                       | `wdth 112`, 700                       |
| Minor (h4)    | 15.5px                     | `wdth 100`, 600                       |
| Body          | 15px                       | 400, lh 1.62, max 72ch                |
| Gauge value   | 27px mono                  | 500, `-.03em`                         |
| Data / meta   | 11–12.5px mono             | 400                                   |

Everything is left-aligned and ragged right. Prose never exceeds 72–74ch.

---

## Structure, not decoration

Rules carry information; they are not trim.

- **Section head** — the title sits on its own 1px ink rule, sentence case. There
  is no tracked-caps label above it. If the section needs a note, it goes *below*
  the title in mono, where it reads as a caption rather than a category.
- **Datum line** — a ruled bar of readings under the hero, cells divided by
  hairlines: run state, stages finished, dataset, model. It replaces the
  middle-dot meta string.
- **Gauge strip** — one band ruled top (2px ink) and bottom (1px ink), the number
  in mono leading and its label following. Not a row of cards. `.gauge.flag`
  switches the top rule and the value to the risk pen.
- **Stage ledger** — numbered rows. Numbering is legitimate here because the
  pipeline genuinely is a sequence of seven stages; a left rule in the pen shows
  which ones the run reached.
- **Annotations** — insights and recommendations hang off a left rule with a
  small mono mark in the gutter, like a reviewer's note in a margin. Graphite for
  a finding, pen for an action, risk for a warning.

### Shape and elevation

`border-radius: 0` on anything that carries data. Panels are sheets lying on a
table: `--sheet` fill, 1px ink border, and a hard offset shadow
`3px 3px 0 rgba(23,28,31,.09)` — no blur, because paper does not glow.

The page ground carries a 28px quadrille at 4.5% ink, so the whole console reads
as ruled stock.

---

## Motion

One orchestrated moment on load, and nothing else uncommanded.

**The plate's entrance** (GSAP, in `ui/assets/pipeline_3d.js`):

1. The bench grid rules itself in and the drawing swings into its axonometric
   view.
2. The rail is struck left to right — each link is a sampled line revealed a
   vertex at a time via `setDrawRange`, so a pen visibly travels it.
3. The seven modules are set down along the rail with a `back.out` stagger.
4. The refinement arc and the RLM tethers are drawn last, being the control flow
   that isn't a straight line.
5. The dataset payload enters ahead of stage 1 and hops forward, inking each
   stage the run actually reached as it arrives.

**The page** gets a single `clip-path` strike on the h1, matched in duration.
Nothing else animates on entry. No per-section fade-and-slide-up, no hover
transitions on every surface.

Everything above is gated on `prefers-reduced-motion: reduce`, which paints the
final state directly. The render loop is additionally gated on an
`IntersectionObserver`, `document.hidden`, and a dirty flag — an idle plate costs
no frames.

---

## The plate

`ui/pipeline_3d.py` owns the state contract; `ui/assets/pipeline_3d.js` draws it.

It is a drawing, so there are **no lights in the scene**. Every surface is a flat
`MeshBasicMaterial` fill in `--sheet` with its edges drawn over the top as
hairlines, the fill pushed back by `polygonOffset` so the edges always win. This
is both the look the brief asks for and the cheapest thing the GPU can do.

State reads as ink pressure, not as new hues:

| Status  | Cage opacity | Fill | Ink        |
| ------- | ------------ | ---- | ---------- |
| pending | .18          | 0    | `graphite` |
| skipped | .12          | 0    | `graphite` |
| done    | .55          | .92  | `pen`      |
| active  | .95          | 1    | `pen`      |
| error   | .85          | .55  | `risk`     |

The core shape of each module is its label — a flat sheet arrives, a crystal
reasons over it, a box does the work, a torus loops, four cubes decompose, two
stacked sheets are the report. The drawing also encodes real control flow: stage
5 arcs back to stage 3, and stage 6 carries two satellites.

**Degradation.** No WebGL context means a plain-text fallback pointing at the
stage ledger, which carries the same information. Pixel ratio is capped at 2 —
hairlines gain nothing above that and low-end GPUs pay for every pixel.

**Access.** The canvas is focusable, labelled `role="img"` with a summary of the
run, and orbits on the arrow keys; `[` and `]` step through the stages and `Home`
recentres. The stage ledger in the page is the non-visual equivalent.

---

## Writing

Plain verbs, sentence case, no filler. Say what happened and what to do.

- Headline: "Every finding, measured twice." — the product's actual
  differentiator, not a category description.
- Errors state the failure and the fix, and do not apologise: "Could not reach
  the model, so the analysis did not start." then what to check.
- Empty states are an invitation: "Nothing on the table yet."
- Buttons name the action and keep that name through the flow.
- No emoji as status. `[done]`, `[run ]`, `[fail]`, `[skip]` in the log; the word
  "risk" where a warning triangle used to be.

---

## Don't

- Add a tracked-out ALL-CAPS label above a heading.
- Round a corner on anything that carries data.
- Use a blurred shadow.
- Use `--risk` for anything other than a result that may not hold.
- Introduce a third pen, or a gradient.
- Animate a section into view on scroll.
- Reach for mono to make a label look technical.
