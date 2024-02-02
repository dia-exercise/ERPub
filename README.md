<div align="center">
    <a href="https://github.com/dia-exercise/ERPub/actions/workflows/ci.yml" rel="nofollow">
        <img src="https://github.com/dia-exercise/ERPub/actions/workflows/ci.yml/badge.svg" alt="CI" />
    </a>
    <a href="https://github.com/dia-exercise/ERPub/blob/main/LICENSE" rel="nofollow">
        <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
    </a>
    <a href="https://github.com/psf/black" rel="nofollow">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
</div>

# ERPub
ERPub is a tool designed for resolving entities across multiple academic publication datasets (specifically ACM and DBLP) by employing various matching functions. This pipeline takes advantage of blocking, matching, and clustering techniques to identify and resolve duplicate entities within the given datasets.

## Installation
Clone this repository to your local machine and navigate to the project directory:
```
git clone https://github.com/dia-exercise/ERPub
cd ERPub
```

Install the required dependencies:
```
pip install -r requirements.txt
```

## Dataset Preparation
The pipeline requires datasets in a specific format. To obtain and prepare the required datasets, you can use the provided script:
```
python data_preparation.py
```
This script downloads the DBLP and ACM datasets, filters publications published between 1995 and 2004, and removes duplicates. The resulting CSV files `DBLP_1995_2004.csv` and `ACM_1995_2004.csv`) will be stored in the `data/prepared` directory.

## Usage
### Importing the Pipeline and it's required functions
```
from erpub.pipeline import pipeline, blocking, matching, preprocessing
```
### Initializing the Pipeline
```
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
```
### Running the Pipeline
```
pipeline.run("output_directory", similarity_threshold=0.8)
```
### Resolving the Entities
```
pipeline.resolve("resolved_output_directory")
```
### Example setup
```
from erpub.pipeline import pipeline, blocking, matching

file_dir = "data/prepared"
pipeline = pipeline.Pipeline(
    file_dir=file_dir,
    blocking_fn=blocking.naive_all_pairs,
    matching_fns={
        "paper_title": matching.jaccard_similarity,
        "author_names": matching.specific_name_matcher,
    },
)

pipeline.run("output_directory", similarity_threshold=0.8)

pipeline.resolve("resolved_output_directory")
```

### Note for vector embeddings
For `matching.vector_embeddings`, the `embeddings_path` of `pipeline.Pipeline` parameter is required. In our case we used the [GloVe](https://github.com/stanfordnlp/GloVe), you can download them to the `embeddings/` directory by running this script:
```
python download_glove_embeddings.py
``` 

## Customization
### Functions
Check out the existing preprocessing, blocking and matching functions in `erpub/pipeline/`. You can also add custom functions to the pipeline on your own. 

### Verbosity
Set the `verbose` parameter of the pipeline to **False** to disable logging.

### Unit Tests
Install pytest for testing:
```
pip install pytest
```
Run unit tests using:
```
pytest tests/
```

## Experiments
Explore various experiments inside [experiments.ipynb](experiments.ipynb) notebook. The notebook provides insights into different use cases and scenarios for applying the entity resolution pipeline.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
