# Tool ideas and future direction

## Category leadership: facet / vector coverage per cluster

Source: LinkedIn post from Blue Array's founder. Captured 2026-06-18.

### The idea
Shift from "keyword leadership" (target a keyword, build a page, fight for the ranking) to
"category leadership": understand a whole category the way a language model does, that is, which
facets or dimensions the models believe matter. The models are shaped by the collective web,
weighted towards authoritative sources such as trusted review sites, so they form a view of what
consumers care about in a category. The opportunity is to find the facets a brand is "underweight"
on relative to what the category cares about.

Example from the post: in the SUV category, models infer that safety is an important facet, because
the collective web says so. Porsche barely talk about safety on their own site, and it hardly
features in their earned media, so on a dimension the models weight heavily, Porsche are underweight,
regardless of how good the cars are. That gap is invisible to keyword thinking. It only shows up when
you look at the category the way a model does.

### Proposed integration into the clustering tool (Nathan's suggestion)
For each top-level cluster (topic), surface the facets / vectors the keywords reveal, that is, the
recurring attributes and dimensions the category cares about. For example, a car insurance topic
surfaces facets like price, comprehensive vs third party, young drivers, comparison, legal cover,
windscreen, no-claims. Weight them by search volume to show what matters most within the category.

### Honest assessment, and the gap to close
- Feasible now, additive: extract facets per cluster from the keywords via the model, weighted by
  volume. This delivers "what the category cares about, by search demand", which sits naturally on top
  of the clustering we already have.
- The deeper point in the post is about what the models believe matters (drawn from authoritative
  sources), which search volume only partly proxies. Some category-important facets, like Porsche's
  "safety", may carry low search volume yet still be model-important, so keyword-derived facets alone
  can miss them.
- To realise the full vision, add two things:
  1. Query the model's own view of the category facets (ask the model what matters in the category),
     not just the keywords.
  2. Measure the brand's coverage of those facets (their site and earned media) to expose the gap,
     the underweight dimensions.
- The keyword facets are a strong first signal. The model-view plus brand-coverage comparison is the
  differentiated, harder part, and the bit that actually produces the Porsche-style insight.

### Suggested phasing
1. Facet extraction per top-level cluster (keywords to weighted facets). Small, fits the current tool.
2. Category facet model: ask the model what facets matter in the category, then merge with the
   keyword facets so low-volume but model-important facets are not missed.
3. Brand coverage gap: assess how well a brand covers each facet across its content, and flag the
   underweight ones.

## Customer reviews as a signal for category facets

Source: Nathan, 2026-06-18. Nathan has a tool that gathers insight from customer reviews at scale.

Reviews are a direct, unprompted source of what customers actually care about in a category, in their
own words. They plug the gap in keyword-only facets: dimensions that are category-important but low
search volume (the Porsche "safety" case) often show up loudly in reviews even when barely searched,
so reviews are arguably a better proxy for "what the category cares about" than volume.

Triangulation: keywords (search demand) + reviews (customer voice) + the model's own view gives a
robust read on category facets, then compared to brand coverage for the gap. Reviews also carry
sentiment, so we learn not just which facets matter but which ones customers praise or complain about,
which keywords cannot tell us.

Open questions: what does the review tool output (raw reviews, themes, sentiment)? Is the data
brand-specific or category-wide across competitors? Both keywords and reviews would need mapping to a
shared facet vocabulary, which the model can reconcile.

## Automated website information architecture (IA)

Source: Nathan, 2026-06-18.

When the core is finished, develop it to output a recommended site information architecture, not just
the page list. The current output is already the skeleton: topics are pillars, and topic-plus-intent
pages are the URLs beneath them. The IA layer would turn that into a buildable structure:
- A hierarchy (pillar, optional sub-pillar, page) as a visual tree or sitemap.
- Suggested URL slugs / paths and a page type per node.
- Internal linking recommendations: pillar pages link down to their pages, pages link back up, and
  related pages within a pillar cross-link. This is the high-value part and is hard to get from
  keyword thinking alone.
- Export as a structured sitemap (CSV) and/or a diagram.

Most of the raw material exists once clustering and intent are done; the IA layer adds hierarchy
depth, slugs, and the linking logic.

## Entities within the pillars

Source: Nathan, 2026-06-18.

Within each pillar, extract the salient entities the topic involves (brands, products, features,
concepts, people, places) from the keywords, and optionally enrich from the ranking pages. Entities
define what a comprehensive pillar should cover; covering the right ones is what builds topical
authority in an entity-led search world.

This dovetails with the category-leadership facets above: facets are the dimensions the category cares
about, entities are the specific things within it. Together they specify what a pillar's content needs
to cover, and support a gap analysis (entities the category or competitors cover that the brand does
not).

## How these fit together (phase 2 vision)

Together these turn the tool from a keyword-to-page map into a category-and-architecture engine:
keywords, then page map, then a recommended IA, with each pillar enriched by its entities and the
facets the category cares about (validated against the model's view and customer reviews, and compared
to the brand's own coverage to expose the gaps).

## Competitor IA as a page-clustering signal

Source: Nathan, 2026-06-18. To try after the semantic + SERP blend is locked.

Let the user name competitor domains and use a strong, well-structured competitor's own page
structure to guide our page boundaries. If a respected editorial competitor (for example the AA) ranks
separate pages for sub-topics within a topic, that is evidence those sub-topics should be separate
pages; if they cover several on one URL, that is evidence to keep them together. Follow a site that has
already won.

It plugs in cheaply because we already pull the top 10 URLs per keyword, so we already know which of a
competitor's URLs rank for which keywords, which is their keyword-to-page mapping. Keywords the
competitor serves with the same URL belong together; keywords served by different URLs split. It
becomes a third signal in the page-clustering blend (semantic + general SERP overlap + competitor
co-occurrence), and a cleaner one, because a good editorial competitor has distinct, intentional pages
rather than the ubiquitous aggregator hubs that muddied generic overlap.

Honest caveats:
- Only gives a signal for keywords the chosen competitor ranks for; fall back to the blend elsewhere.
- The user must pick strong editorial competitors (category leaders), not aggregators, or it adds noise.
- It is a prior, not gospel: their IA can be imperfect, and sometimes you will want to out-structure
  them to win, so it should be a weighted signal rather than an override.
- Multiple competitors may disagree; aggregate or let the user set a primary.

## Relevance check against the target domain

Source: Nathan, 2026-06-18.

The user names the domain they are building the clustering for. Once the pillars and pages are built,
the tool flags any clusters that are clearly irrelevant to that domain, so off-topic noise (common in
scraped or multi-source keyword lists) can be pruned.

How it would work: derive a short description of what the domain offers (from the domain, or by reading
its homepage), then have the model score each pillar's relevance to that scope and flag the clearly-off
ones. Present them as flags for the user to confirm, never auto-delete (consistent with not removing
anything without a check).

Caveats: relevance is a judgement, and some clusters are adjacent rather than off-topic (a car insurer
may legitimately want "courtesy car" content), so flag only the clearly irrelevant ones and leave the
decision to the user. It is the inverse of the category-leadership and brand-coverage ideas: those say
what the brand should cover, this prunes what it should not.

## User-defined entities as a cross-cutting filter

Source: Nathan, 2026-06-18.

Let users supply their own entities, the business-relevant dimensions they care about, and use them as
an extra filter and grouping axis on top of Topic > Pillar > Page. Example: a travel client adds a list
of countries, then filters the whole map to "Spain" to see every topic, pillar and page that relates to
Spain, cutting across the hierarchy.

The hierarchy answers "how should this content be structured". A user entity answers "show me the slice
of it that matters to my business". The two are orthogonal, so this sits on top of the existing output
rather than changing it.

How it would work:
- The user provides the entity list (countries, brands, product ranges, audiences). This can be seeded
  from the "entities within the pillars" extraction above and confirmed by the user, or typed in directly.
- Tag each keyword with the entities it relates to. Explicit mentions (a country or brand named in the
  keyword) are reliable by matching; implicit ones (a city implying its country, a landmark implying a
  destination) need the model or a reference list.
- A keyword can carry zero, one, or several entities, so this is tagging, not partitioning. The
  Topic > Pillar > Page hierarchy stays a clean split; entities are tags layered over it.
- Surface it as a filter in the front-end and as extra columns in the export, with an "unassigned"
  bucket for keywords that match no entity.

It also opens up an entity-by-topic coverage view: for each entity (each country, say), how much of the
topic and page structure it covers and at what search volume, which highlights where coverage is thin.

Honest assessment and caveats:
- Additive and low-risk: it annotates and filters, and does not touch the clustering logic.
- Explicit entities are easy; implicit ones are harder and need the model or a gazetteer, with some error.
- Plenty of keywords name no entity at all ("best time to visit"), so the unassigned bucket is essential;
  do not force-fit them.
- It overlaps with "Entities within the pillars": that extracts the entities a topic involves, this lets
  the user pick the ones that matter and navigate by them. They can share one entity vocabulary.

## Named runs and projects

Source: Nathan, 2026-06-19.

Today a run is identified only by a timestamp (the run_id, e.g. 20260619-103040), and the "open a
previous run" control in step 1 is a flat list of those timestamps. As the tool gets used across clients
and keyword sets, that list becomes hard to navigate. Two improvements:

- Let the user give a run a title when they run it (for example "RAC car insurance, UK, June"), stored
  with the run and shown wherever runs are listed, so past runs are recognisable at a glance rather than
  by timestamp.
- Let the user group runs into projects (typically a client or a site), so previous runs are organised
  and easy to come back to. The previous-runs picker becomes project, then run, rather than one long
  flat list.

How it would work:
- Add a title (free text) and a project (pick an existing one or create a new one) to the Run step,
  before "Run mapping".
- Persist both with the run. The data branch already stores each run under `runs/<run_id>/`; add the
  title and project to the run's `result.json` / `status.json` meta, and optionally namespace the path
  as `runs/<project>/<run_id>/` so a project's runs sit together. The SERP cache stays shared (it is
  keyed by location + keyword), so projects do not duplicate SERP spend on overlapping keywords.
- In the front-end, replace the flat previous-runs selectbox with a project picker then a run picker,
  each run shown by its title and date. A light projects view could list projects with their run counts.

Honest assessment and caveats:
- Mostly a front-end and metadata change; it does not touch the clustering or intent logic, so it is
  low-risk and additive.
- Keep it backwards compatible: existing timestamp-only runs should still load (treat them as an
  "untitled" / "no project" bucket).
- Decide whether a project is just a label on runs or a first-class object. The label approach is
  simpler and probably enough to start.
- If runs are re-pathed under a project, keep reading the old `runs/<run_id>/` location too, or migrate,
  so nothing already saved is lost.

## Map page clusters to existing client pages, and a competitor overview

Source: Nathan, 2026-06-19.

For a client domain, map each recommended page (cluster) to the existing URL on the client's site that
already ranks for the majority of that cluster's keywords. The output then says "you already have a page
for this, here it is, improve it" rather than implying everything is new, and the pages with no matching
client URL stand out as the genuine content gaps.

It plugs in cheaply because we already pull the top 10 ranking URLs per keyword. For each page cluster,
tally how often a client URL appears across its keywords and attach the one that covers the majority,
with the share (for example "ranks for 24 of 31 keywords") and an average position.

Add a competitor overview alongside it: for named competitor domains, show which of their URLs rank for
each cluster's keywords and where (position), so the client sees who owns each page-level topic and how
they have structured it. It uses the same SERP data and is the reporting counterpart to the "Competitor
IA as a page-clustering signal" idea (that one uses competitor co-occurrence to influence the clustering;
this one just reports it).

How it would work:
- Client domain (and optional competitor domains) entered up front. The relevance-check idea already
  wants the client domain, so they can share that input.
- For each page, tally client and competitor URLs across the member keywords' top-10 results. Attach the
  dominant client URL as the "existing page", with coverage share and average position, else mark the
  page as "new / gap".
- Output per page: existing client URL, coverage (X of Y keywords), average position, plus a competitor
  matrix (which competitor ranks, their top URL, average position).

Caveats:
- "Majority of keywords" needs a sensible threshold. Surface the share so the user can judge rather than
  hard-deciding for them.
- A client may rank one URL across several recommended pages (a sign to consolidate) or several URLs for
  one page (a sign they have overlap to tidy). Both are useful, so report the mapping rather than forcing
  it to be one-to-one.
- It only sees keywords where the domain ranks in the top 10. Below that, this data cannot show it.

## Column mapping and extra input data

Source: Nathan, 2026-06-18 (requested in the earlier session, recovered 2026-06-19).

Let the user upload a CSV with whatever columns they have and map them to the tool's inputs, rather than
relying on fixed column-name detection. A short mapping step after upload (this column is the keyword,
this is volume, this is difficulty, and so on) makes the tool work with an export from any source (Ahrefs,
Semrush, Search Console) without re-formatting first.

It also opens the door to extra input dimensions, such as keyword difficulty, CPC or current position,
which can then enrich the output and the prioritisation (for example flag low-difficulty pages, or weight
pages by opportunity rather than volume alone).

- Fits the Upload step: once a file is read, show the detected columns and let the user assign each to a
  role, remembering sensible defaults.
- Keyword is required; volume and the rest are optional. Any extra columns flow through to the page and
  keyword exports.

## Visual output: charts and graphs

Source: Nathan, 2026-06-18 (recovered 2026-06-19).

Make the results far more visual rather than tables alone: a Topic > Pillar > Page tree or treemap, a
volume-by-topic chart, an intent split, and a page-count or opportunity view. This helps a non-technical
user read the structure at a glance and makes the output more presentable to a client. It sits on top of
the existing output, since the data is already there.

## Spend controls: per-run and monthly caps

Source: Nathan, 2026-06-18 (recovered 2026-06-19).

Add guardrails on cost: a maximum keywords-per-run limit and a total monthly keyword limit, so a large or
accidental upload cannot run up an unexpected DataForSEO bill. Warn and require confirmation when an
upload exceeds the per-run cap, and track usage against the monthly cap. This pairs naturally with showing
the estimated SERP cost before the Run step.

## More visual run / progress screen

Source: Nathan, 2026-06-19.

The run-in-progress screen (step 3) is currently a plain info box ("Running... 112s elapsed") and a log
link. Make it more visually appealing and more informative:
- A branded progress state: a progress bar or spinner in the palette, a clear elapsed timer, and the
  stepper showing step 3 as active.
- Real progress rather than just elapsed seconds. The SERP fetch is the long pole and the engine already
  logs "collected X of Y SERPs", so a bar driven by that count is the most useful single readout. A stage
  checklist (embedding, reading SERPs, clustering, naming pages, done) advancing through the run would
  read well too.
- Once a fraction of the SERPs are back, project a rough time remaining.

How it would work:
- Have the engine write progress into `status.json` each poll cycle (it already writes `status.json` at
  the start and logs the collected count), then have the front-end render it as a bar plus stage list.
- Replace the single `st.info` with that bar, stage list and timer, styled in the brand palette.

Related:
- This is the in-progress counterpart to "Visual output: charts and graphs" (which covers the results).
- Worth pairing with persisting the run id in the URL query params, so a reload (or a front-end redeploy)
  resumes the progress view instead of dropping the user back to step 1. Today the run continues
  server-side, but the in-app view is held only in session state, so a refresh loses it.

## Email when a run finishes

Source: Nathan, 2026-06-19.

Let the user add their email at the Run step and get emailed when the run completes, with a link straight
to the result, so they do not have to keep the tab open or remember to check back. This fits the
server-side model: the run already survives the tab closing, and an email removes the need to watch it at
all, which matters most for the long runs that the higher caps now allow.

How it would work:
- Optional email field on the Run step, stored with the run (in its input or meta on the data branch).
- On completion, send the email. Either the Action sends it as a final step (it already knows when the
  run finished and can build the result link) or a small notify step does. Needs an email service and a
  secret (a transactional email API, or SMTP), added like the existing OpenAI and DataForSEO secrets.
- Send on failure too, so a failed run is not a silent dead end.

Caveats:
- Needs an email-sending service and credentials; a simple transactional provider is enough.
- User emails are personal data. If the repo is public, do not write them in plaintext to the runs
  branch; keep them out of committed files or store only a short-lived reference.
- Pairs with "named runs and projects" (the email can use the run's title) and the progress-screen idea.

## Scale to large keyword sets (30k+): resume mode and scalable clustering

Source: Nathan, 2026-06-19.

The SERP side is now resumable and lossless: SERPs are cached as they arrive, a cancelled run still
saves its cache, and DataForSEO's `id_list` lets already-paid tasks be re-fetched for free within 30
days. That makes large runs viable in principle, but two ceilings stop a single 30k run today.

1. Collection time vs the runner limit. Collecting 14k on the cheap queue took about 3 hours; 30k is
   roughly 5 to 7 hours, and a GitHub-hosted job is hard-capped at 6 hours. So 30k probably will not
   collect inside one job on the cheap queue.
2. Clustering memory. The pillar step runs one agglomerative clustering over every keyword's embedding,
   which builds a full pairwise matrix (order n squared). At 30k that is roughly 4 to 7GB, around a
   standard runner's RAM, so it risks running out of memory. This is independent of the queue and is the
   harder blocker.

What to build:
- Resume / chunk mode. Let a run pick up where it left off: collect what fits in a job, cache it, and on
  the next run fetch the rest. Already-submitted tasks come back free via `id_list` within 30 days, so
  no re-paying. Either automate the resume, or split a large upload into chunks that each fit a job, with
  the shared cache stitching them together.
- Memory-scalable clustering. Replace or augment the single n-squared agglomerative pass for very large
  sets. Options: reduce dimensions first (UMAP) then cluster with HDBSCAN (the method this repo was named
  after), use mini-batch or blockwise clustering, or run the clustering step on a larger paid runner with
  more RAM only when the set is big.
- A higher DataForSEO priority stays the lever if speed matters more than cost (faster collection, about
  twice the per-SERP price).

Caveats:
- Cost scales linearly: about $18 of SERP for 30k on the cheap queue, plus a little OpenAI.
- The resume flow leans on the cache and `id_list` recovery being solid, so they should be part of the
  normal path, not just a rescue route.

## Singleton handling: keep, fold, or quarantine (relevance-gated)

Source: Nathan, 2026-06-22.

On a large long-tail set the page step leaves many single-keyword pages (1,946 of 3,411 pages, 57%, on
the 14,183-keyword crowdfunding test). Most are zero or near-zero volume long-tail, a few are genuine
niche queries, and some are outright noise (typos, URLs, off-topic headlines). The fix is a
relevance-gated three-way split rather than blunt merging:

1. Keep: a singleton whose volume is above a small floor stays as its own page (a real standalone term).
2. Fold: a singleton that is semantically close to an existing page (cosine to the nearest page centroid
   above a threshold) is folded into that page. The closeness gate is what stops unrelated keywords
   polluting a good page.
3. Quarantine: a singleton that is not close to any page goes into an "ungrouped / for review" bucket,
   never merged into a page and never auto-deleted. The user decides whether to keep or bin those.

Validated on the 14k set (volume floor 100, similarity 0.55): keep 2, fold 1,922, quarantine 22, taking
pages from 3,411 to 1,467 with every keyword still accounted for. Folds landed correctly (for example
"crowd funding sec" into "Crowdfunding Regulations Explained"), and quarantine caught exactly the noise
(a stray "Keyword" header, an immigration query, news headlines, a TV show). Keyword Insights produced
5,408 clusters on the same set, so the tool is more consolidated even before folding, and much more after.

How it would work:
- A post-clustering step after pages are formed. Reuse the keyword embeddings already computed, build a
  centroid per multi-keyword page, score each singleton against the nearest page, then apply the volume
  floor and similarity threshold.
- Expose the volume floor and similarity threshold as advanced settings, and add an "ungrouped / review"
  section to the output and the app so quarantined keywords stay visible.

Caveats:
- "Irrelevant vs niche-but-relevant" is a judgement and the threshold is a proxy. Mistyped but relevant
  queries (for example "colection pot") can score low and land in quarantine, which is why it is a review
  list, not auto-delete.
- The source-level fix is density clustering (HDBSCAN, the repo's namesake): it labels noise during
  clustering, giving the quarantine bucket for free rather than as a bolt-on. Pairs with the
  "scale to large keyword sets" idea.

## Split oversized pillars (cap the mega-pillar)

Source: Nathan, 2026-06-22.

A broad head term can pull a large share of the set into one pillar: on the 14k crowdfunding test
"Crowdfunding Essentials" held 219 pages / 2,563 keywords (15% of the set), the same pattern as the
"comprehensive" mega-pillar on the car-insurance set. Keyword Insights has it worse (its "crowd funding"
hub holds 1,415 clusters), but it is still a flaw. The 0.45 pillar threshold is loose enough that
everything semantically near the head term lands together.

Two ways to address it:
- Global: a tighter pillar threshold splits the broad terms, but it also fragments every other pillar and
  raises the total pillar count, so it is a blunt lever.
- Targeted: keep the 0.45 default, then re-cluster only the pillars above a size cap at a tighter
  threshold, so the mega-pillar splits into sub-pillars while the rest of the structure is untouched.
  This is the cleaner fix.

How it would work:
- After pillar clustering, flag any pillar over a cap (for example > 300 keywords or > 40 pages).
- Re-cluster each flagged pillar's members at a tighter threshold (recursively until under the cap),
  relabel the resulting sub-pillars, and keep them under the same topic.
- Expose the cap and the sub-threshold as advanced settings.

Caveats:
- The right cap and sub-threshold need validating per set, so expose them rather than hard-coding.
- Some breadth is genuine, so the split should produce named, meaningful sub-pillars rather than arbitrary
  fragments; label them after splitting.
- Relates to the scalable-clustering idea: HDBSCAN varies density per region instead of using one global
  threshold, which would handle this more naturally.
