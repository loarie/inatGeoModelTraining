import numpy as np
import tensorflow as tf


class FocalScatterAdd(tf.keras.layers.Layer):
    """Adds a (batch, n_focal) tensor into specific class columns of a (batch, num_classes)
    tensor, leaving all other columns unchanged. Used to inject a small dedicated focal-species
    head's logits on top of the shared backbone's logits at exactly the focal class indices."""

    def __init__(self, num_classes, indices, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.indices = list(indices)
        scatter = np.zeros((len(self.indices), num_classes), dtype=np.float32)
        for i, cls in enumerate(self.indices):
            scatter[i, cls] = 1.0
        self.scatter_matrix = tf.constant(scatter)

    def call(self, inputs):
        base, focal_logits = inputs
        return base + tf.matmul(focal_logits, self.scatter_matrix)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "indices": self.indices})
        return config


class IndependentPerClassHead(tf.keras.layers.Layer):
    """Gives each of num_classes an entirely INDEPENDENT small 2-layer MLP applied to the same
    raw input - not a shared hidden layer serving all classes (that's what caused the dead-class
    collapse in make_geo_model_net_focal_head at N=500 (v33/v34) and N=5,007 (v30): every class
    started at an identical or randomly-perturbed point in a bottleneck shared by every other
    class, and enough of them lost the resulting competition to get stuck at 0 forever). Here
    there is NOTHING shared between classes in this pathway - no bottleneck to compete for, no
    symmetric start to race from, regardless of how N or the hidden width are chosen. The cost
    is parameter count instead of instability: N independent tiny networks costs roughly
    N * (input_dim*hidden_dim + hidden_dim) parameters, modest at realistic hidden widths
    (e.g. ~1,100 params/class at hidden_dim=16) even at N in the hundreds of thousands.

    Vectorized via batched matmul (einsum) rather than building N separate Dense layers, which
    would be far too slow to construct/trace in TF at real scale.

    First layer (W1/b1): normal (he_normal-equivalent) init - doesn't matter for symmetry since
    every class's weights are already independent regardless of init scheme, but the hidden
    representation needs non-zero weights to be useful. Second/output layer (W2/b2):
    zero-initialized, so the whole head contributes exactly 0 for every class at epoch 0 - the
    same deliberate no-op start as the proven-good v28 recipe, growing influence only as
    training finds it useful.
    """

    def __init__(self, num_classes, input_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        he_std = (2.0 / self.input_dim) ** 0.5
        self.W1 = self.add_weight(
            name="W1", shape=(self.num_classes, self.input_dim, self.hidden_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=he_std), trainable=True,
        )
        self.b1 = self.add_weight(
            name="b1", shape=(self.num_classes, self.hidden_dim),
            initializer="zeros", trainable=True,
        )
        self.W2 = self.add_weight(
            name="W2", shape=(self.num_classes, self.hidden_dim),
            initializer="zeros", trainable=True,
        )
        self.b2 = self.add_weight(
            name="b2", shape=(self.num_classes,),
            initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        h = tf.einsum("bi,nih->bnh", inputs, self.W1) + self.b1[tf.newaxis, :, :]
        h = tf.nn.relu(h)
        out = tf.einsum("bnh,nh->bn", h, self.W2) + self.b2[tf.newaxis, :]
        return out  # (batch, num_classes) - one independent scalar per class per example

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes, "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
        })
        return config


class ResLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=256):
        super(ResLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.w1 = tf.keras.layers.Dense(
            hidden_dim, activation="relu", kernel_initializer="he_normal"
        )
        self.w2 = tf.keras.layers.Dense(
            hidden_dim, activation="relu", kernel_initializer="he_normal"
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.w1(inputs)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.add([x, inputs])
        return x

    def get_config(self):
        return {"hidden_dim": self.hidden_dim}


def make_geo_model_net(num_classes, num_input_feats=5, hidden_dim=256):
    fcnet = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(
                shape=(num_input_feats,),
            ),
            tf.keras.layers.Dense(
                hidden_dim, activation="relu", kernel_initializer="he_normal"
            ),
            ResLayer(hidden_dim),
            ResLayer(hidden_dim),
            ResLayer(hidden_dim),
            ResLayer(hidden_dim),
            tf.keras.layers.Dense(num_classes, use_bias=False),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Activation("sigmoid", dtype="float32", name="predictions"),
        ]
    )
    return fcnet


def make_geo_model_net_focal_head(
    num_classes, focal_class_indices, num_input_feats=5, hidden_dim=256, focal_hidden_dim=64,
    zero_init_final_layer=True,
):
    """Same shared backbone as make_geo_model_net (all 10,007+ classes get logits from it,
    unchanged), plus a small SEPARATE head for focal_class_indices only. The focal head takes
    the raw input directly (not the shared hidden_dim representation), so learning a subtle
    covariate signal for one sparse focal species doesn't have to compete for gradient budget
    in the backbone shared by every other class. The two branches' logits are summed before the
    final sigmoid. Added after v23-v27 (5 loss-side "more absence pressure" mechanisms, 6
    training runs) all regressed vs v22 - see focal_10k_config_v26.yml/v27.yml for that history.
    The diagnosis behind this: Millipede's coastal range has a real, near-zero-overlap bio4
    (temperature seasonality) signature vs. the Rocky Mountains false-positive band, but the
    model wasn't learning to use it - a capacity/attention problem (one 512-dim hidden layer
    shared across 10,007 classes, only 13,206 Millipede presence points to compete for it), not
    a loss-weighting problem, so the fix is architectural rather than another loss term.

    zero_init_final_layer: when True (default, matches v28's proven-good 7-species recipe -
    NEVER change this default), the final focal Dense layer is zero-initialized so the head
    starts as a no-op and only grows influence as training finds it useful. When False (tested
    at N=500 in v34 after v30/v33 both showed ~50-85% of a large focal group's classes go
    permanently dead - sigmoid saturation with zero gradient - by a few epochs in), every one of
    N classes instead starts at an IDENTICAL zero point and must break symmetry via gradient
    descent from there; the hypothesis is this symmetric race, not just raw N-vs-hidden_dim
    crowding, is what actually kills classes at scale (v22's plain backbone has an almost
    identical crowding ratio to v30's failed head - 10,007/512 vs 5,007/256, both ~19.6 - yet
    doesn't fail, and its final layer uses normal random init, not zero). False switches the
    final Dense layer to the SAME he_normal init every other layer in this network already uses,
    so classes start already-differentiated instead of racing from a shared degenerate point.
    """
    inputs = tf.keras.layers.Input(shape=(num_input_feats,))

    x = tf.keras.layers.Dense(hidden_dim, activation="relu", kernel_initializer="he_normal")(inputs)
    x = ResLayer(hidden_dim)(x)
    x = ResLayer(hidden_dim)(x)
    x = ResLayer(hidden_dim)(x)
    x = ResLayer(hidden_dim)(x)
    shared_logits = tf.keras.layers.Dense(num_classes, use_bias=False)(x)

    f = tf.keras.layers.Dense(focal_hidden_dim, activation="relu", kernel_initializer="he_normal")(inputs)
    f = tf.keras.layers.Dense(focal_hidden_dim, activation="relu", kernel_initializer="he_normal")(f)
    focal_logits = tf.keras.layers.Dense(
        len(focal_class_indices), use_bias=True,
        kernel_initializer=("zeros" if zero_init_final_layer else "he_normal"),
    )(f)

    combined_logits = FocalScatterAdd(num_classes, focal_class_indices)([shared_logits, focal_logits])
    combined_logits = tf.keras.layers.Dropout(rate=0.2)(combined_logits)
    outputs = tf.keras.layers.Activation("sigmoid", dtype="float32", name="predictions")(combined_logits)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def make_geo_model_net_independent_head(
    num_classes, focal_class_indices, num_input_feats=5, hidden_dim=256, independent_head_dim=16
):
    """Same shared backbone as make_geo_model_net (all classes get logits from it, unchanged),
    plus a small extra head for focal_class_indices only - but unlike
    make_geo_model_net_focal_head, this head has ZERO weight-sharing between the focal classes.
    Each one gets its own fully independent 2-layer MLP (see IndependentPerClassHead) rather
    than all of them reading out through one shared hidden bottleneck.

    Added after make_geo_model_net_focal_head's shared-bottleneck design was confirmed (twice:
    v30 at N=5,007, v33/v34 at N=500, with both zero-init and random-init on the final layer)
    to let a large fraction of focal classes get permanently stuck at near-zero output - a
    capacity-competition/symmetric-race failure that a shared bottleneck is inherently exposed
    to regardless of width or init scheme. Removing all sharing removes the failure mode by
    construction: there's nothing for classes to compete over or start identically from.
    """
    inputs = tf.keras.layers.Input(shape=(num_input_feats,))

    x = tf.keras.layers.Dense(hidden_dim, activation="relu", kernel_initializer="he_normal")(inputs)
    x = ResLayer(hidden_dim)(x)
    x = ResLayer(hidden_dim)(x)
    x = ResLayer(hidden_dim)(x)
    x = ResLayer(hidden_dim)(x)
    shared_logits = tf.keras.layers.Dense(num_classes, use_bias=False)(x)

    focal_logits = IndependentPerClassHead(
        num_classes=len(focal_class_indices), input_dim=num_input_feats, hidden_dim=independent_head_dim,
    )(inputs)

    combined_logits = FocalScatterAdd(num_classes, focal_class_indices)([shared_logits, focal_logits])
    combined_logits = tf.keras.layers.Dropout(rate=0.2)(combined_logits)
    outputs = tf.keras.layers.Activation("sigmoid", dtype="float32", name="predictions")(combined_logits)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
