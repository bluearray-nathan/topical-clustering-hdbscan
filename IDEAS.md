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
