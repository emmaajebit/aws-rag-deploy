import pytest

from sentiment.data.loaders import load_reviews
from sentiment.preprocessing import prepare_dataset
from sentiment.train_ml import run_experiment


@pytest.fixture(scope="session")
def small_df():
    return prepare_dataset(load_reviews("synthetic", n=3000, seed=7))


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, small_df):
    out = tmp_path_factory.mktemp("artifacts")
    run_experiment(small_df, models=["lr", "nb"], cv=3, out_dir=out, fig_dir=out / "fig", verbose=False)
    return out / "best_model"
