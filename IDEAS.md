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
