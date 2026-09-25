# unitsim

Ground-truth simulation for unit discovery in animal vocal sequences.

Structure claims about animal vocalisations -- that song units follow Zipf's law, that frequent units
are short -- are made on units a pipeline recovered, not units anyone observed. `unitsim` generates
communication systems whose units, frequencies and structure are known, runs published unit-discovery
pipelines over them, and scores what comes back.

- `generate` -- lexicons with a chosen frequency law and brevity, and streams with ground-truth boundaries
- `corrupt` -- element-label confusion, over-splitting and under-splitting
- `metrics` -- boundary / token / lexicon P-R-F, cluster agreement, and the published log-log R^2
- `powerlaw_fit` -- discrete power-law MLE, KS xmin selection, bootstrap goodness of fit, and a
  likelihood-ratio test against a lognormal (Clauset, Shalizi & Newman 2009)

MIT licence.
