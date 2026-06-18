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
