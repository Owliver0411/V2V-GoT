import torch
import torch.nn as nn
import re
# `re` = Python's regular-expression module, used below to parse a
# projector-type string like "mlp2x_gelu" and pull the number "2" out of it.

# =============================================================================
# BIG PICTURE / WHAT THIS FILE DOES
# =============================================================================
# This is the file that actually DEFINES what a "projector" (the translator
# network mentioned throughout the other files) looks like on the inside.
# Recall: a projector's job is to take a feature vector of one size (coming
# from the vision tower, or from this project's sensor-data processing) and
# convert it into a vector of a DIFFERENT size -- specifically, the size the
# language model uses for its own word embeddings (config.hidden_size, e.g.
# 4096) -- so the two kinds of vectors can be mixed together in one sequence.
#
# This file offers a few different possible projector "shapes", selected by
# a string setting called `mm_projector_type`:
#   - 'linear':       the simplest possible option -- a single matrix
#                      multiplication (one "layer") and nothing else.
#   - 'mlpNx_gelu':    a small stack of N linear layers with GELU
#                      "activation functions" in between (explained below).
#                      Bigger N = more layers = the network can learn more
#                      complex transformations, at the cost of more
#                      parameters/compute.
#   - 'identity':      a "do-nothing" pass-through option, useful mainly for
#                      testing/debugging or when the input is already the
#                      right size.
#
# There are TWO builder functions here:
#   - build_vision_projector: builds the regular projector (mm_projector),
#     used for image patches, "shallow"-mode scene features, and all
#     object-level features (see the earlier llava_arch.py comments).
#   - build_scene_vision_projector: this project's own custom addition,
#     builds a SEPARATE, second projector (mm_scene_projector) used only for
#     "deep"-mode scene-level features, because those come in with a
#     different input vector size (e.g. 3072) than everything else (e.g.
#     1024), so they need their own dedicated network sized to match.
# =============================================================================


class IdentityMap(nn.Module):
    # PLAIN-LANGUAGE EXPLANATION:
    # `nn.Module` is PyTorch's base class for "a piece of a neural network"
    # -- anything from a single layer to an entire model inherits from it.
    # This particular one is a network that does absolutely NOTHING to its
    # input -- it just hands back exactly what it was given, unchanged. It
    # exists so that "no projector at all" can be handled with the exact
    # same code interface (i.e. "call it like a function and get a tensor
    # back") as the real projectors below, rather than needing special-case
    # `if projector is None` checks scattered everywhere else in the
    # codebase.
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # `forward` is the standard PyTorch method name for "what happens
        # when you actually run data through this network" -- calling
        # `some_module(x)` automatically calls `some_module.forward(x)`
        # under the hood. `*args, **kwargs` just means "silently accept and
        # ignore any extra arguments passed in", so this can be swapped in
        # anywhere a real projector is expected without breaking due to
        # mismatched function signatures.
        return x

    @property
    def config(self):
        # A `@property` lets you access this like a plain attribute
        # (`identity_map_instance.config`) instead of having to call it
        # like a function (`identity_map_instance.config()`). This fake
        # "config" dictionary exists purely so that other code which
        # expects every projector to have a `.config` describing its type
        # (e.g. for logging/saving purposes) doesn't crash when handed an
        # IdentityMap instead of a real trainable projector.
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    # PLAIN-LANGUAGE EXPLANATION:
    # A "residual block" ("Res" = residual) is a small, well-known building
    # block in modern neural network design. The core idea: instead of a
    # layer directly computing "the new output", it computes "how much
    # should we ADJUST the input by", and then that adjustment gets ADDED
    # back onto the original input (see the `x + self.proj(x)` line in
    # `forward` below). This tends to make deep networks much easier to
    # train, because even if the "adjustment" part learns something totally
    # unhelpful early in training, the original input still passes straight
    # through mostly unchanged (rather than potentially being erased by a
    # badly-initialized layer) -- like a safety net that keeps information
    # flowing even before the adjustment layers have learned anything
    # useful.
    #
    # NOTE: this class is defined here but doesn't actually appear to be
    # used anywhere else in this file (build_vision_projector and
    # build_scene_vision_projector never reference SimpleResBlock) -- it
    # looks like a leftover/unused building block, possibly kept around
    # from an earlier experiment or copied from the original LLaVA
    # codebase for potential future use.
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        # "LayerNorm" (Layer Normalization) is a very common neural-network
        # technique that rescales a vector's numbers so they have a
        # consistent, well-behaved statistical range (roughly zero average,
        # unit spread) before further processing. This tends to make
        # training faster and more stable, similar in spirit to the
        # "normalize distances by dividing by 200 meters" trick seen in the
        # earlier llava_arch.py file, but LayerNorm does this automatically
        # and adaptively (with a small number of its own learnable
        # adjustment numbers) rather than using one fixed manual constant.
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            # `nn.Linear(in, out)` is the fundamental "translator" building
            # block used throughout this whole file: it's just a matrix
            # multiplication (plus an addable "bias" number) that turns a
            # vector of `in` numbers into a vector of `out` numbers. Here
            # `in` and `out` are the same (`channels`), so this particular
            # linear layer doesn't change the vector's SIZE, only its
            # actual number values.
            nn.GELU(),
            # "GELU" (Gaussian Error Linear Unit) is an "activation
            # function" -- a small, simple mathematical curve applied to
            # every number in a vector, one at a time. Activation functions
            # are essential in neural networks because without them, no
            # matter how many `nn.Linear` layers you stack, the whole thing
            # mathematically collapses into being equivalent to just ONE
            # big linear layer (since stacking linear transformations
            # without anything else in between doesn't add any new
            # expressive power). Inserting a non-linear activation function
            # like GELU between linear layers is what allows a multi-layer
            # network to actually learn more complex, curved
            # relationships instead of only straight-line ones.
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
        # This is the "residual"/"skip connection" pattern described above:
        # take the (normalized) input `x`, compute an adjustment via the
        # small linear-GELU-linear stack (`self.proj(x)`), and add that
        # adjustment back onto `x` itself, rather than replacing `x`
        # entirely with the adjustment's output.


def build_vision_projector(config, delay_load=False, **kwargs):
    # PLAIN-LANGUAGE EXPLANATION:
    # This is a "factory function" -- instead of you manually writing
    # `nn.Linear(...)` or `nn.Sequential(...)` yourself every time you need
    # a projector, you just call this function with a config object, and it
    # constructs and returns whichever kind of projector network the config
    # asks for. `delay_load` and `**kwargs` are accepted for interface
    # consistency with other builder functions in this codebase (like
    # build_vision_tower) but aren't actually used inside this particular
    # function's body.
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    # Read the desired projector type string out of the config (e.g.
    # "linear", "mlp2x_gelu", "identity"), defaulting to "linear" if the
    # config doesn't specify one at all.

    if projector_type == 'linear':
        # The simplest option: ONE single linear layer, going straight from
        # the vision tower's feature size (config.mm_hidden_size, e.g.
        # 1024) to the language model's embedding size (config.hidden_size,
        # e.g. 4096). This is literally just one matrix multiplication --
        # no activation functions, no extra layers. It's the "translator"
        # in its most stripped-down form: just a straight linear
        # re-scaling/re-mixing of the numbers into a new size.
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    # This regular expression looks for strings that match the exact
    # pattern "mlp" + one-or-more digits + "x_gelu", such as "mlp2x_gelu" or
    # "mlp3x_gelu". "MLP" stands for "Multi-Layer Perceptron" -- the classic
    # term for a neural network built from a stack of plain linear layers
    # with activation functions in between (exactly the kind of network
    # being built in the next block). If the projector_type string matches
    # this pattern, `mlp_gelu_match` will be a "match object" you can pull
    # the captured digits out of; if it doesn't match, `mlp_gelu_match` will
    # be `None`.
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        # `.group(1)` extracts the digits captured by `(\d+)` in the regex
        # above -- e.g. for "mlp2x_gelu", this gives the string "2", which
        # gets converted to the integer 2. This number is how many linear
        # layers ("depth") the resulting network will have in total.
        # MY_DEBUG
        # only change the input size of mm_projector
        #modules = [nn.Linear(1024, config.hidden_size)]
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        # Start the list of network layers with ONE linear layer that
        # converts from the vision/sensor feature size straight to the
        # language model's embedding size. Every layer after this first one
        # will then be size hidden_size -> hidden_size (see below), since
        # by this point the vector is already the "right" size and just
        # needs further refining, not further resizing.
        for _ in range(1, mlp_depth):
            # This loop runs (mlp_depth - 1) more times. E.g. for
            # "mlp2x_gelu" (mlp_depth=2), the loop runs once, adding one
            # more GELU + Linear pair, for a total network of: Linear ->
            # GELU -> Linear (2 linear layers total, matching the "2" in
            # "mlp2x").
            modules.append(nn.GELU())
            # Insert a GELU activation function BEFORE each additional
            # linear layer -- this is what makes stacking multiple linear
            # layers actually meaningful/more powerful than a single one
            # (see the detailed GELU explanation inside SimpleResBlock
            # above).
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
        # `nn.Sequential(*modules)` chains all these layers together into
        # one single network object: running data through it means running
        # it through each layer in the list, one after another, in order
        # (Linear -> GELU -> Linear -> GELU -> Linear -> ...). The `*`
        # unpacks the Python list into individual arguments, since
        # `nn.Sequential` expects the layers as separate arguments rather
        # than as one list object.

    if projector_type == 'identity':
        # "No real projector at all" option -- just pass features through
        # unchanged. Useful mainly if the vision/sensor features already
        # happen to be exactly the right size, or for quick
        # testing/debugging without training a real projector.
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
    # If projector_type doesn't match any of the recognized options above,
    # fail loudly with a clear error message rather than silently doing
    # something wrong or unexpected.


# MY_CODE
# create another projector for scene level feature
def build_scene_vision_projector(config, delay_load=False, **kwargs):
    # PLAIN-LANGUAGE EXPLANATION:
    # This is essentially a near-identical COPY of build_vision_projector
    # above, but hard-wired to read its INPUT size from a different config
    # field (`mm_scene_projector_input_size` instead of `mm_hidden_size`).
    # This is the function that builds `mm_scene_projector`, the dedicated
    # second translator network used only for "deep"-mode scene-level
    # bird's-eye-view features (which come in as 3072-number vectors,
    # unlike everything else in this project which gets padded/shaped to
    # 1024 numbers before going through the regular mm_projector). Having a
    # totally separate network (rather than trying to reuse mm_projector
    # for both) means this scene projector can have its OWN dedicated set
    # of learned numbers specifically tuned for translating this particular
    # kind of feature, without being forced to also handle the other,
    # differently-structured inputs.
    #mm_scene_projector_input_size = 1024
    #mm_scene_projector_input_size = 320
    #mm_scene_projector_input_size = 3072
    # (These commented-out lines show earlier experiments/attempts where
    # this input size was hardcoded directly to different values -- 1024,
    # then 320, then 3072 -- before settling on reading it dynamically from
    # the config instead, which is more flexible since it can be set
    # per-experiment without editing this file.)
    mm_scene_projector_input_size = config.mm_scene_projector_input_size
    print('mm_scene_projector_input_size: ', mm_scene_projector_input_size)
    #assert False
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    # NOTE: this reuses the SAME `mm_projector_type` config setting as the
    # regular projector -- meaning whatever architecture choice ('linear',
    # 'mlpNx_gelu', or 'identity') is picked, it applies to BOTH
    # mm_projector and mm_scene_projector at once; there's no separate
    # config field to give the scene projector a different number of
    # layers than the regular one.
    if projector_type == 'linear':
        return nn.Linear(mm_scene_projector_input_size, config.hidden_size)
        # Same "single matrix multiplication" idea as before, just using
        # the scene-feature input size (e.g. 3072) instead of
        # config.mm_hidden_size.
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        # MY_DEBUG
        # only change the input size of mm_projector
        #modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        modules = [nn.Linear(mm_scene_projector_input_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
        # Identical multi-layer-with-GELU construction to
        # build_vision_projector's version, just starting from a different
        # input vector size.
    if projector_type == 'identity':
        return IdentityMap()
    raise ValueError(f'Unknown projector type: {projector_type}')
