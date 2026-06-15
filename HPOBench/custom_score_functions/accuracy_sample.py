def sample_accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    """Individual accuracy scores for each sample.

    Similar to sklearn's accuracy_score but returns a list of individual scores
    for each prediction instead of the overall average.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, default=True
        If ``False``, return individual scores as integers (1 or 0).
        Otherwise, return individual scores as floats (1.0 or 0.0).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If provided, individual scores are weighted.

    Returns
    -------
    scores : list of float or int
        Individual accuracy scores for each prediction.
        - 1 (or 1.0) if prediction is correct
        - 0 (or 0.0) if prediction is incorrect

    Examples
    --------
    >>> y_true = [0, 1, 2, 2, 1]
    >>> y_pred = [0, 1, 1, 2, 1]
    >>> accuracy_sampled(y_true, y_pred)
    [1.0, 1.0, 0.0, 1.0, 1.0]

    >>> accuracy_sampled(y_true, y_pred, normalize=False)
    [1, 1, 0, 1, 1]

    In the multilabel case with binary label indicators:

    >>> accuracy_sampled([[0, 1], [1, 1]], [[0, 1], [1, 0]])
    [1.0, 0.0]
    """

    # Convert inputs to lists for consistent handling
    def _to_list(arr):
        """Convert array-like input to list."""
        if hasattr(arr, "tolist"):
            return arr.tolist()
        elif hasattr(arr, "__iter__") and not isinstance(arr, str):
            return list(arr)
        else:
            return [arr]

    y_true = _to_list(y_true)
    y_pred = _to_list(y_pred)

    # Check input consistency
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Input arrays have inconsistent lengths: {len(y_true)} != {len(y_pred)}"
        )

    if sample_weight is not None:
        sample_weight = _to_list(sample_weight)
        if len(sample_weight) != len(y_true):
            raise ValueError(
                f"Sample weights have inconsistent length: {len(sample_weight)} != {len(y_true)}"
            )

    # Detect if this is multilabel classification
    def _is_multilabel(y):
        """Check if the input represents multilabel data."""
        if not y:
            return False
        # Check if first element is array-like (list, tuple, etc.)
        first_element = y[0]
        return (
            hasattr(first_element, "__iter__")
            and not isinstance(first_element, str)
            and not isinstance(first_element, (int, float))
        )

    is_multilabel = _is_multilabel(y_true) or _is_multilabel(y_pred)

    # Compute individual accuracy scores
    scores = []

    if is_multilabel:
        # Multilabel case: exact match required for each sample
        for i in range(len(y_true)):
            true_labels = _to_list(y_true[i])
            pred_labels = _to_list(y_pred[i])

            # Check if lengths match
            if len(true_labels) != len(pred_labels):
                score = 0  # Different lengths = incorrect
            else:
                # Check if all labels match exactly
                score = 1 if true_labels == pred_labels else 0

            scores.append(score)
    else:
        # Single-label case: direct comparison
        for i in range(len(y_true)):
            score = 1 if y_true[i] == y_pred[i] else 0
            scores.append(score)

    # Apply sample weights if provided
    if sample_weight is not None:
        scores = [score * weight for score, weight in zip(scores, sample_weight)]

    # Convert to appropriate type based on normalize parameter
    if normalize:
        scores = [float(score) for score in scores]
    else:
        scores = [int(score) for score in scores]

    return scores
