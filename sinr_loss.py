import tensorflow as tf


def sinr_loss(y_true, logits, logits_bg, pos_weight, effort_bg_matrix=None, alpha=1.0, bg_base_weight=1.0,
              bg_effort=None, bg_present=None, focal_cols=None, bg_range_penalty=None):
    """
    SINR loss: positive examples + random background locations.

    Operates on raw (pre-sigmoid) logits via tf.nn.softplus's numerically-stable log(1+exp(.)),
    NOT on sigmoid probabilities via -log(prob+eps) (the prior formulation). That prior version
    has a real trap: once a class's predicted probability underflows below the 1e-5 eps - which
    happens well before the logit reaches anywhere near float32's actual underflow point -
    loss_pos flattens to a near-constant (the eps swamps the true, smaller probability) and stops
    contributing gradient, while loss_bg's per-example gradient also shrinks toward 0, leaving no
    restoring force to pull a runaway class back - only the still-large background pool's
    shrinking-but-nonzero pull in the SAME direction. Diagnosed in v35 (see
    focal_10k_config_v35.yml): with zero cross-class weight sharing
    (IndependentPerClassHead) ruling out the old shared-bottleneck collapse, roughly half of a
    500-species focal group still ran away to permanently-saturated negative logits (down to -50
    by epoch 4, still drifting, never recovering) while the rest converged cleanly - traced to
    exactly this vanishing-gradient floor. softplus keeps a bounded, non-vanishing gradient
    (saturating at -1 for loss_pos, 0->1 for loss_bg) regardless of how wrong the current
    prediction is, so a class can climb back out even from a very wrong starting point.

    loss_pos: only the true species at each presence location is pushed toward 1.
              Other species are NOT pushed down here — the bg loss handles that.
              This avoids diluting the positive signal by K (number of species).

    loss_bg:  all K species at each background location are pushed toward 0, with
              a flat bg_base_weight applied everywhere regardless of effort or
              distance from a species' own range. If effort_bg_matrix is provided,
              extra absence penalty on top of that floor is scaled by
              effort_scale * effort[j] * (1 - species_present[j,k]):
                - species present in cell j  → weight 0 (no extra penalty)
                - species absent, high effort → strong extra absence signal
                - species absent, low effort  → floor only (bg_base_weight)

    effort_bg_matrix: optional float32 numpy array shape (B_bg, K).

    Alternative to effort_bg_matrix (same loss) that never builds a (B_bg, K) weight matrix on
    the host - at ~176k classes that matrix is ~360MB per step:
      bg_effort:        (B_bg,) effort_scale * effort[j], applied to every species at row j.
      bg_present:       (B_bg, K) 1 where the class IS present in row j's H3 cell - its effort
                        weight is subtracted back out.
      focal_cols:       (F,) class indices that get a range penalty.
      bg_range_penalty: (B_bg, F) extra absence weight for those columns.
    Pass either effort_bg_matrix or these arguments, not both.
    bg_base_weight: flat absence-pressure floor applied to every background point,
              independent of effort or range-penalty. This is the lever for
              "lower intensity where there's no presence, regardless of sampling."
    pos_weight: float32 tensor of shape (num_classes,) - one weight per class, gathered per
              example by its true class. A uniform vector (every entry equal) reproduces the
              old flat-scalar behavior exactly; a non-uniform vector lets classes with very
              different observation counts get different positive-loss weight (see
              train_tf_sinr_test.py's --pos_weight_balance_exponent, added after v30's
              5,007-species universal-head run diverged - a single flat pos_weight applied to
              classes ranging from 100 to 100,000 real observations let the highest-count
              classes' aggregate positive-loss gradient dominate and destabilize training).
    """
    inds = tf.range(tf.shape(y_true)[0], dtype=tf.int64)
    inds = tf.stack([inds, y_true], axis=1)
    pos_logits = tf.gather_nd(logits, inds)
    per_example_pos_weight = tf.gather(pos_weight, y_true)
    loss_pos = per_example_pos_weight * tf.nn.softplus(-pos_logits)  # -log(sigmoid(x)) = softplus(-x)

    loss_bg = tf.nn.softplus(logits_bg)  # -log(1-sigmoid(x)) = softplus(x)
    if bg_effort is not None or bg_range_penalty is not None:
        # sum of bg_weight * loss_bg, decomposed: flat floor + per-row effort over every class,
        # minus the effort at present (row, class) pairs, plus the range penalty on focal columns
        weight_row = bg_base_weight + (bg_effort if bg_effort is not None else 0.0)
        total_bg = tf.reduce_sum(tf.reshape(weight_row, [-1, 1]) * loss_bg)
        if bg_effort is not None and bg_present is not None:
            total_bg -= tf.reduce_sum(bg_present * loss_bg * tf.reshape(bg_effort, [-1, 1]))
        if bg_range_penalty is not None:
            total_bg += tf.reduce_sum(tf.gather(loss_bg, focal_cols, axis=1) * bg_range_penalty)
        return tf.reduce_mean(loss_pos) + total_bg / tf.cast(tf.size(loss_bg), total_bg.dtype)

    if effort_bg_matrix is not None:
        bg_weight = bg_base_weight + effort_bg_matrix
    else:
        bg_weight = bg_base_weight

    loss_bg = bg_weight * loss_bg

    return tf.reduce_mean(loss_pos) + tf.reduce_mean(loss_bg)


@tf.function
def neg_log(x):
    return -tf.math.log(x + 1e-5)


def sinr_loss_legacy(y_true, yhat, yhat_bg, pos_weight, alpha=1.0):
    """
    Original SINR loss on sigmoid probabilities, kept unchanged for train_tf_sinr.py.
    sinr_loss above is the logit-based replacement used by train_tf_sinr_test.py.
    Args:
        y_true: True labels
        yhat: Predictions from the model for 'x'
        yhat_bg: Predictions from the model for bg/fake samples
        pos_weight: Scaling factor for positive examples
        alpha: Weight for bg loss
    """

    loss_pos = neg_log(1.0 - yhat)
    inds = tf.range(tf.shape(y_true)[0], dtype=tf.int64)
    inds = tf.stack([inds, y_true], axis=1)
    pos_preds = tf.gather_nd(yhat, inds)

    newvals = pos_weight * neg_log(pos_preds)
    loss_pos = tf.tensor_scatter_nd_update(loss_pos, [inds], [newvals])

    loss_bg = neg_log(1.0 - yhat_bg)

    return tf.reduce_mean(loss_pos) + tf.reduce_mean(loss_bg)
