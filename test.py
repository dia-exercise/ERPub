from erpub.pipeline import pipeline, blocking, matching, preprocessing

file_dir = "data/prepared"
pipeline = pipeline.Pipeline(
    file_dir=file_dir,
    preprocess_data_fn=preprocessing.all_lowercase_and_stripped,
    blocking_fn=blocking.same_year_of_publication,
    matching_fns={
        "paper_title": matching.jaccard_similarity,
        "author_names": matching.specific_name_matcher,
    },
)
