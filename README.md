# IMDb Sentiment Analysis: Traditional ML + PyTorch NLP

This project compares traditional machine-learning algorithms and
neural-network approaches for classifying IMDb movie reviews as
**positive** or **negative**.

## Experiments included

-   **Traditional ML:** Logistic Regression, Linear SVM, Random Forest,
    and Histogram-based Gradient Boosting.
-   **Text features:** Binary word presence, word/ngram counts (1--3
    grams), and TF-IDF.
-   **Sentence embeddings:** A pretrained
    `sentence-transformers/all-MiniLM-L6-v2` encoder with a PyTorch
    dense neural-network classifier.
-   **Sequence models:** Single-layer and double-layer bidirectional
    LSTM (BiLSTM) classifiers.
-   **Optional GloVe embeddings:** Code is included, but the download
    and training commands are commented out by default.
-   **BERT:** Fine-tuning the compact pretrained model
    `prajjwal1/bert-mini`.
-   **Inference:** Predict sentiment for a new review using the
    fine-tuned BERT model.

> **Note:** The sentence-embedding experiment uses Sentence-Transformers
> as a comparable alternative to TensorFlow Hub NNLM/Universal Sentence
> Encoder; it is not an exact reproduction of those original embedding
> models. Tree-based experiments use reduced features and a subset of
> the data to manage runtime and memory, so their scores are not
> directly comparable with every other experiment.

## Requirements

-   Python 3.10 or newer is a reasonable starting point for a fresh
    environment.
-   Jupyter Notebook or JupyterLab.
-   Internet access to download Python packages, the IMDb dataset, and
    pretrained model files.
-   An NVIDIA GPU with a working CUDA-enabled PyTorch installation is
    recommended for faster neural-model training. CPU execution may work
    but can be considerably slower, especially for BERT.

The notebook installs these packages in its first setup cell:

-   `datasets`
-   `transformers`
-   `sentence-transformers`
-   `scikit-learn`
-   `nltk`
-   `matplotlib`
-   `pandas`
-   `tqdm`

PyTorch (`torch`) is also required. Install a PyTorch build appropriate
for your operating system and hardware by following the official
instructions: https://pytorch.org/get-started/locally/

## Setup

### 1. Get the project files

Place `imdb_sentiment_analysis_pytorch.ipynb` in your project folder.
Save this README as `README.md` in the same folder.

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**

``` powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

If PowerShell blocks activation, you can run the notebook using the
environment's Python executable directly, or follow your organization's
policy for script execution.

**Linux / macOS**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install PyTorch using the instructions at
https://pytorch.org/get-started/locally/ for your system. Then install
the remaining packages:

``` bash
python -m pip install --upgrade pip
python -m pip install datasets transformers sentence-transformers scikit-learn nltk matplotlib pandas tqdm jupyter
```

The notebook's first code cell also installs its listed dependencies. If
you have already installed them in the environment, you can skip that
cell.

### 4. Start Jupyter

``` bash
jupyter notebook
```

Or, if you use JupyterLab:

``` bash
jupyter lab
```

Open `imdb_sentiment_analysis_pytorch.ipynb`.

## Run the notebook

Run the cells **in order from top to bottom**. Later cells depend on
variables and models created earlier.

1.  **Install dependencies** --- install any missing libraries.
2.  **Imports and device setup** --- imports packages, sets random
    seeds, and selects CUDA or CPU.
3.  **Load IMDb data** --- downloads the dataset and creates training,
    validation, and test splits.
4.  **Text cleaning** --- prepares text for the classical ML and LSTM
    experiments.
5.  **Classical ML baselines** --- trains and evaluates the traditional
    classifiers and plots their validation accuracy.
6.  **Sentence-embedding MLP** --- generates sentence embeddings, trains
    the dense classifier, and reports validation/test accuracy.
7.  **LSTM preparation** --- creates the vocabulary, token IDs,
    datasets, and data loaders.
8.  **BiLSTM models** --- trains and evaluates single-layer and
    double-layer BiLSTMs.
9.  **Optional GloVe** --- skip by default. The download and training
    lines are commented out; enable them only if you want to run this
    additional experiment.
10. **BERT setup and fine-tuning** --- loads BERT-mini, tokenizes the
    reviews, fine-tunes the classifier, and evaluates it.
11. **Consolidated results** --- creates the results table and
    neural-model accuracy chart.
12. **Sample prediction** --- runs sentiment prediction on a new review
    using BERT.

The notebook's section numbering and code-cell boundaries may differ
slightly from this checklist; use the section headings in the notebook
as your guide.

## Dataset splits

The notebook loads the IMDb dataset through Hugging Face Datasets. It
uses the official training split of 25,000 reviews and divides it into:

-   **Training:** 20,000 reviews
-   **Validation:** 5,000 reviews
-   **Test:** 25,000 reviews from the official test split

The test split should be reserved for final evaluation. Use validation
performance to compare models during development.

Labels are encoded as `0` for negative sentiment and `1` for positive
sentiment.

## Hardware and runtime tips

These experiments can require substantial time and memory. For an
initial smoke test:

-   Confirm that imports and dataset loading work before launching every
    model.
-   Start with a smaller subset of the data or fewer epochs when
    debugging.
-   Reduce the BERT maximum sequence length (`MAX_LEN_BERT`) or batch
    size (`BERT_BATCH_SIZE`) if GPU memory is insufficient.
-   Reduce embedding batch size (`EMBED_BATCH_SIZE`) if generating
    sentence embeddings runs out of memory.
-   Skip the optional GloVe download unless you specifically want to run
    that experiment.
-   Tree-based models already use reduced features and data subsets in
    this notebook.

If you change the dataset size or training configuration, record those
changes when comparing results.

## Results and interpretation

The notebook produces a classical-model results table, a consolidated
results table, and a chart comparing neural models on test accuracy.

Keep these evaluation details in mind:

-   Classical linear models report validation and test accuracy.
-   Tree-based models report validation accuracy on a subset and leave
    test accuracy blank.
-   Neural models report validation and test accuracy.
-   Accuracy alone does not show which classes are misclassified.
    Precision, recall, F1-score, and a confusion matrix would provide a
    more complete evaluation.

Do not assume that one model is best before running the notebook; report
the values actually produced in your environment.

## Predict sentiment for a new review

The final section defines `predict_sentiment_bert(text)`. After the BERT
setup and fine-tuning cells have completed, run a prediction such as:

``` python
predict_sentiment_bert(
    "The acting was excellent and the story was surprisingly moving."
)
```

The function returns the predicted sentiment and the model's negative
and positive probabilities. These probabilities reflect the model's
output and do not guarantee that a prediction is correct.

## Troubleshooting

### `ModuleNotFoundError`

Install the missing package in the same Python environment used by
Jupyter. If you installed a package after starting the notebook, restart
the kernel if necessary.

### PyTorch does not detect the GPU

Check your PyTorch installation and CUDA compatibility using the
official PyTorch installation guide. The notebook falls back to CPU when
CUDA is unavailable.

### Out-of-memory errors

Reduce batch sizes and sequence lengths, or run fewer examples while
testing the pipeline. Sentence embedding generation and BERT fine-tuning
can be memory-intensive.

### The notebook is taking too long

Run a small subset first, reduce epochs for a smoke test, and skip the
optional GloVe experiment. Full runs of the classical tree models and
BERT may take substantial time.

### Results differ between runs

The notebook sets random seeds, but exact reproducibility is not
guaranteed across all hardware, library versions, and GPU operations.
Record your environment and configuration when reporting results.

## Project structure

``` text
project-folder/
├── README.md
└── imdb_sentiment_analysis_pytorch.ipynb
```

## Acknowledgements

-   IMDb movie-review dataset, accessed through Hugging Face Datasets.
-   Hugging Face Transformers and the pretrained `prajjwal1/bert-mini`
    model.
-   Sentence-Transformers and `all-MiniLM-L6-v2`.
-   Scikit-learn and PyTorch.
-   GloVe is an optional embedding resource and is not downloaded or
    trained by default in this notebook.
