from amie.config import ModelName

class ModelUnavailableError(RuntimeError):
    """A selected model dependency could not be loaded."""


def classifier(name: ModelName, seed: int):
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=100, max_depth=5,
            min_samples_leaf=5, random_state=seed, n_jobs=1)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError, ValueError) as exc:
            raise ModelUnavailableError(
                "XGBoost could not load. On macOS install OpenMP with "
                "`brew install libomp`, then retry. On other platforms check "
                "the XGBoost native runtime installation."
            ) from exc

        return XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
            subsample=1, colsample_bytree=1, random_state=seed, n_jobs=1,
            eval_metric="logloss", tree_method="hist")
    raise ValueError("Unsupported model")
