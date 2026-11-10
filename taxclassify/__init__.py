"""taxclassify: classify NTSB reports into the buckets of a grounded taxonomy version (OBJ-3, stage 1).

Separate from taxground: it reads a taxground run folder (grounding_results.json, inputs/, taxonomy.json) or a release
folder, and needs no database. Stage 1 model: per-bucket calibrated logistic regression within each CICTT parent,
split-conformal prediction sets for abstaining, a verbatim evidence sentence and an explanation per prediction.
"""
__version__ = "0.1.0"
