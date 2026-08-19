#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed und  er the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# =============================================================================
# BIG PICTURE / WHAT THIS FILE DOES
# =============================================================================
# This file is part of "LLaVA" (Large Language and Vision Assistant), a system
# that lets a text-only language model (like Vicuna/LLaMA) "see" and reason
# about non-text data (in this case, camera images AND autonomous-driving
# sensor data like point clouds / bird's-eye-view maps from vehicles).
#
# The core trick used across all "vision-language models" like LLaVA is:
#   1. Take some non-text data (an image, a sensor map, etc).
#   2. Run it through a small neural network that turns it into a sequence of
#      numeric vectors that "look like" the word/token embeddings the language
#      model already understands. This small converter network is called a
#      "projector" (think of it as a translator that turns pictures/sensor
#      data into a foreign language: the language model's internal numeric
#      vocabulary).
#   3. Splice those converted vectors into the sequence of word embeddings
#      right where a special placeholder token (like "<image>") appears in
#      the prompt.
#   4. Feed the combined sequence (text embeddings + inserted "vision
#      embeddings") into the language model like normal. The language model
#      doesn't know or care that some of its "words" actually came from a
#      camera or sensor -- it just sees a sequence of vectors.
#
# This particular file has been HEAVILY modified from the original open-source
# LLaVA project to work with autonomous-driving cooperative-perception data:
# multiple connected vehicles (CAVs = "Connected Autonomous Vehicles") each
# see the same scene from different positions, and the code combines their
# sensor-derived features ("scene-level" bird's-eye-view features and
# "object-level" detected-bounding-box features) into a big sequence of
# tokens that gets fed into the language model, similar to how image patches
# would be fed in for a normal vision-language model.
#
# Two classes are defined here:
#   - LlavaMetaModel: sets up and stores the "vision tower" (the network
#     that reads images) and the "projector" networks (the translators
#     mentioned above).
#   - LlavaMetaForCausalLM: an abstract mixin class with the actual logic for
#     turning images/sensor-data into token embeddings and splicing them into
#     the text sequence before it goes into the language model.
# =============================================================================


from abc import ABC, abstractmethod
# ABC = "Abstract Base Class". A class inheriting from ABC cannot be
# instantiated directly -- it exists only to be inherited from by other
# classes that fill in the missing ("abstract") pieces. Think of it as a
# blueprint/contract: "any class that wants to be a LlavaMetaForCausalLM
# MUST implement get_model()".

import torch
import torch.nn as nn
# PyTorch is the numerical/deep-learning library used here. "torch.Tensor" is
# basically a multi-dimensional array of numbers (like a numpy array) that
# can live on a GPU and remembers how it was computed so gradients can be
# calculated automatically during training.
# "torch.nn" contains building blocks for neural networks: layers,
# parameters (learnable numbers), activation functions, etc.

from .multimodal_encoder.builder import build_vision_tower
# A "vision tower" is the neural network that turns a raw image (a grid of
# pixel values) into a set of numeric feature vectors describing what's in
# the image. This is usually a pretrained model like CLIP's image encoder.
# "build_vision_tower" is a factory function: give it a config, and it
# constructs (or loads) the right vision network for you.

from .multimodal_projector.builder import build_vision_projector, build_scene_vision_projector
# The "projector" is a small neural network (often just one or two linear
# layers, i.e. a simple matrix multiplication + optional activation) whose
# only job is to change the *size* (dimensionality) of a feature vector so
# it matches whatever size the language model expects for its token
# embeddings. E.g. the vision tower might output 1024-number vectors, but
# the language model expects 4096-number vectors per token -- the projector
# is what converts 1024 numbers into 4096 numbers.
# There are two separate projectors here:
#   - mm_projector: used for regular image features and "object-level"
#     detection features.
#   - mm_scene_projector: a second, separate projector specifically for the
#     "deep" scene-level bird's-eye-view sensor features (they have a
#     different input size, so they need their own translator network).

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# IGNORE_INDEX: a special numeric label (usually -100) used to tell the
#   training loss function "don't penalize the model for what it predicts at
#   this position" -- used for padding and for the inserted image/sensor
#   tokens (since there's no "correct next word" to predict there).
# IMAGE_TOKEN_INDEX: a special placeholder token ID that stands for "an
#   image goes here" inside the text prompt. Wherever this ID shows up in
#   the tokenized text, the code below will cut it out and glue in the real
#   image/sensor feature vectors instead.
# DEFAULT_IMAGE_PATCH_TOKEN / DEFAULT_IM_START_TOKEN / DEFAULT_IM_END_TOKEN:
#   literal text strings (like "<im_patch>", "<im_start>", "<im_end>") that
#   get added to the tokenizer's vocabulary so the model can be taught to
#   recognize where images begin/end in text form.

from llava.mm_utils import get_anyres_image_grid_shape
# Helper function used only in the "any resolution" image path (splitting a
# large image into a grid of smaller sub-images/patches so very
# high-resolution pictures can still be handled). Not used in the
# point-cloud / sensor-data path that this project actually exercises.

import math
# Standard Python math library, used later for computing positional
# encodings (see get_positional_embedding below).


class LlavaMetaModel:
    # This class is meant to be mixed into (inherited alongside) the actual
    # underlying language model class (e.g. LlamaModel). It doesn't define a
    # full model by itself -- it just adds the "vision-related" pieces
    # (vision tower + projector networks) on top of whatever base language
    # model it's combined with.

    def __init__(self, config):
        # "config" is an object (similar to a dictionary) holding all the
        # settings for this model: sizes of layers, which vision tower to
        # use, etc. It usually comes from a Hugging Face `AutoConfig`.
        super(LlavaMetaModel, self).__init__(config)
        # Calls the constructor of whatever class this gets mixed with
        # (e.g. the base LLaMA model), passing along the config so the
        # normal language-model machinery gets set up first.

        if hasattr(config, "mm_vision_tower"):
            # "mm" = "multimodal". If the config says which vision tower to
            # use, it means this checkpoint is meant to handle images (or,
            # in this project, sensor data), so we build the extra modules.
            print('config: ', config) # here

            # Build (construct, but don't yet load real weights into) the
            # vision tower network. delay_load=True means "set up the
            # architecture now, but don't actually download/load the heavy
            # pretrained weights until later" -- this saves time/memory when
            # you don't need the vision tower yet (e.g. while just
            # inspecting the config).
            self.vision_tower = build_vision_tower(config, delay_load=True)

            # Build the two "translator" networks described above.
            self.mm_projector = build_vision_projector(config)
            self.mm_scene_projector = build_scene_vision_projector(config)

            # "requires_grad = True" means: during training, please compute
            # gradients (i.e. "how should this number change to reduce the
            # error") for this parameter, and update it. Setting it
            # explicitly to True here guards against the projector
            # accidentally being frozen (left un-trainable) by some other
            # part of the code, such as when LoRA fine-tuning is used (LoRA
            # normally freezes most of the model and only trains small
            # add-on matrices, so this makes sure the projector isn't
            # accidentally left frozen).
            for p in self.mm_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            for p in self.mm_scene_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            #self.initialize_mm_scene_projector()

            # (Duplicate of the loop above -- looks like leftover debugging
            # code that repeats the same "make sure these are trainable"
            # step twice. Harmless, just redundant.)
            for p in self.mm_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            for p in self.mm_scene_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
            #assert False

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                # This branch relates to a technique ("unpad") for handling
                # non-square, high-resolution images by removing padding
                # that was added to make the image square before patchifying
                # it. It is NOT used in this project (the assert False below
                # proves this code path is intentionally never triggered
                # here), since this project mostly deals with sensor data,
                # not high-res images.
                print('here unpad')
                # not hit
                assert False
                # "image_newline" would be a single learnable vector added
                # between rows of image patches, similar to a newline
                # character, so the language model can tell where one row of
                # image patches ends and the next begins.
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )


    # MY_CODE
    # ------------------------------------------------------------------
    # This method is a leftover/experimental helper (currently not called
    # anywhere active -- it's commented out below where it would be used,
    # and it ends in "assert False" so it would deliberately crash if run).
    # Its purpose: load a previously-trained "mm_scene_projector" from a
    # saved checkpoint file on disk, so you don't have to retrain it from
    # scratch every time.
    # ------------------------------------------------------------------
    def initialize_mm_scene_projector(self):
        # https://github.com/eddyhkchiu/my_co_llm_driver/blob/bb7af2486e61886311454ec186005e5dad0f2d87/LLaVA/llava/model/builder.py#L76

        # torch.load reads a saved dictionary of tensors from disk. A
        # ".bin" checkpoint file for a LoRA fine-tune typically contains
        # only the *extra* trainable weights (here, the projector weights),
        # not the whole giant language model, to keep the file small.
        # map_location='cpu' means "load these tensors onto the CPU" (as
        # opposed to directly onto a GPU), which is safer if you don't know
        # what GPU is available.
        non_lora_trainables = torch.load('checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v6sdup3_init_scene_and_object/non_lora_trainables.bin', map_location='cpu')

        # The saved dictionary's keys are strings like
        # "base_model.model.mm_scene_projector.0.weight" -- these prefixes
        # come from how the model was wrapped during training (e.g. with
        # PEFT/LoRA wrappers). The next lines strip off those wrapper
        # prefixes so the keys match the plain parameter names this
        # projector module expects (e.g. just "0.weight").
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}

        #if any(k.startswith('model.model.') for k in non_lora_trainables):
        #    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

        non_lora_trainables = {(k[31:] if k.startswith('model.model.mm_scene_projector.') else k): v for k, v in non_lora_trainables.items()}

        print('non_lora_trainables: ', non_lora_trainables)

        # load_state_dict copies the loaded numbers into the actual
        # projector network's weights. strict=False means "don't complain
        # if some keys are missing or extra -- just load whatever matches".
        self.mm_scene_projector.load_state_dict(non_lora_trainables, strict=False)
        assert False  # deliberately stops execution here; this function is not meant to run in normal operation right now.


    def get_vision_tower(self):
        # Small "getter" helper. In some distributed-training setups
        # (specifically FSDP, "Fully Sharded Data Parallel"), the vision
        # tower gets wrapped in a length-1 Python list to prevent the
        # training framework from trying to shard/split it across GPUs (the
        # vision tower is usually frozen/not trained, so it doesn't need
        # that treatment). This function unwraps it if needed, so callers
        # elsewhere in the code don't have to worry about which form it's
        # in.
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # This function is called once, typically at the start of training,
        # to actually set up (or re-configure) the vision tower and
        # projector based on command-line/training arguments, as opposed to
        # __init__ above which sets things up based on a saved model
        # config. Both paths exist because a model can either be started
        # fresh from a base language model (no vision parts yet -- this
        # function creates them) or resumed from an already-multimodal
        # LLaVA checkpoint (parts already exist -- __init__ handles that).
        #print('model_args: ', model_args)

        vision_tower = model_args.vision_tower
        # e.g. the name of a pretrained CLIP model to use as the image
        # encoder, such as "openai/clip-vit-large-patch14-336".
        mm_vision_select_layer = model_args.mm_vision_select_layer
        # Vision towers like CLIP have many internal layers. This setting
        # picks WHICH internal layer's output to use as "the image
        # features" (often a layer near, but not exactly at, the very end,
        # since the very last layer's features are sometimes too
        # specialized for CLIP's own training task).
        mm_vision_select_feature = model_args.mm_vision_select_feature
        # Vision transformers often produce one extra summary vector (a
        # "[CLS] token", short for "classification token", which
        # summarizes the whole image) plus one vector per image patch. This
        # setting picks whether to keep just the patch vectors ("patch") or
        # include the summary vector too ("cls_patch").
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        # Optional path to a separately pretrained projector checkpoint (an
        # "MLP adapter" = "Multi-Layer Perceptron adapter", i.e. the
        # projector, since it's often built from one or more simple fully
        # connected/"linear" layers) to load instead of starting the
        # projector from random numbers.
        mm_patch_merge_type = model_args.mm_patch_merge_type
        # Controls how multiple image patches get combined/arranged (e.g.
        # "flat" = just list them one after another; "spatial_unpad" = keep
        # track of their 2-D row/column layout). Not really exercised in
        # this project's sensor-data path.

        self.config.mm_vision_tower = vision_tower
        # Record the chosen vision tower's name into the saved config so
        # that if this model is saved and reloaded later, it knows which
        # vision tower to rebuild.

        if self.get_vision_tower() is None:
            # No vision tower exists yet -- build a brand new one from
            # scratch (this is the "starting fresh from a plain language
            # model" scenario).
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                # If using FSDP (a technique for splitting a huge model's
                # weights across multiple GPUs to save memory), wrap the
                # vision tower in a list as a trick to keep FSDP from trying
                # to shard it (see get_vision_tower's comment above).
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            # A vision tower object already exists (e.g. reloading a
            # checkpoint) -- just make sure its real weights are loaded
            # into memory (load_model actually pulls in the pretrained
            # numbers, since earlier construction may have used
            # delay_load=True to skip that step).
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        # Record in the config: "yes, this model uses a multimodal
        # projector".
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        # What kind of projector to build -- e.g. "linear" (a single matrix
        # multiplication, the simplest possible translator) versus more
        # complex options like a small multi-layer network with
        # non-linearities (activation functions) in between.
        self.config.mm_hidden_size = vision_tower.hidden_size
        # The size (number of numbers per vector) that the vision tower
        # outputs -- this becomes the projector's INPUT size.
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            print('here 1')
            # train from vicuna checkpoint will hit here
            # No projector exists yet -- this is the "starting from a plain
            # language model, no vision parts at all" case. Build a brand
            # new projector network. Its output size is automatically set
            # to match the language model's own embedding size
            # (self.config.hidden_size), since that's what it needs to
            # produce so its output can be mixed in with normal word
            # embeddings.
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                # See the explanation of "image_newline" above -- this is
                # the same idea, just built here instead of in __init__ for
                # the "starting fresh" scenario. Not used by this project.
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
            #assert False
        else:
            print('here 2')
            # hit here when train from llava checkpoint
            # A projector already exists (loaded from a previous LLaVA
            # checkpoint). Force its parameters (and the scene projector's)
            # to be trainable again -- this matters specifically when using
            # LoRA fine-tuning, because loading a LoRA-wrapped checkpoint
            # can accidentally leave everything (including the projector)
            # frozen (requires_grad = False) by default, and we want the
            # projector to keep learning.
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            for p in self.mm_scene_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            #self.initialize_mm_scene_projector()
            # does not work
            # torch.Size([0])
            #assert False

        if pretrain_mm_mlp_adapter is not None:
            # If a path to a separately pretrained projector was given,
            # load it in, replacing whatever the projector currently has.
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                # Filters the loaded checkpoint dictionary down to only the
                # entries whose key contains `keyword` (e.g. "mm_projector"),
                # and strips the prefix so the remaining key names match
                # exactly what mm_projector.load_state_dict expects.
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print('here: pretrain_mm_mlp_adapter')
            # so far no hit
            assert False
            # This whole block is effectively "dead code" for this project
            # right now (the assert False proves it's never actually
            # reached in current usage) -- it exists in case someone wants
            # to reuse a projector pretrained in a separate, earlier stage.


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    # PLAIN-LANGUAGE EXPLANATION:
    # When you resize a non-square photo to fit into a square box (which
    # many vision models require), you usually add blank padding stripes on
    # the top/bottom or left/right so the picture doesn't get stretched or
    # squished. This function figures out how much of that padding was
    # added and removes it, so the leftover tensor matches the picture's
    # true (original) aspect ratio again. "CxHxW" means the tensor's shape
    # is [number of Channels (e.g. 3 for RGB), Height, Width].
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        # The original image was relatively wider than the padded version
        # -- meaning padding was added along the height (top and bottom).
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        # Padding was added along the width instead (left and right).
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    # "CausalLM" = "Causal Language Model", i.e. a model that predicts the
    # next word given all the previous words (it can only look "backwards"
    # at what came before, never "forwards" -- that's what "causal" means
    # here, similar to cause always coming before effect).
    #
    # This class is a "mixin": it's combined with an actual language model
    # class (like LlamaForCausalLM) to add vision/sensor-data capabilities
    # on top of normal text generation. It requires whatever it's mixed
    # into to implement get_model() (see the @abstractmethod below) so this
    # class can reach into the underlying model's vision tower and
    # projectors.

    @abstractmethod
    def get_model(self):
        # This method has no implementation here on purpose -- it's a
        # placeholder that the class this gets mixed into MUST provide,
        # returning the underlying LlavaMetaModel instance (defined above)
        # so this class can access self.get_model().vision_tower,
        # self.get_model().mm_projector, etc.
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        # Turns a batch of raw images into a batch of token-like feature
        # vectors ready to be spliced into the language model's input.
        #print('images.shape: ', images.shape)
        # [1, 3, 336, 336]
        # Shape meaning: [batch_size=1 image, 3 color channels (RGB),
        # 336 pixels tall, 336 pixels wide].

        # Step 1: run the image through the vision tower (e.g. CLIP's image
        # encoder). This breaks the image into a grid of small square
        # "patches" (like cutting a photo into a checkerboard of tiles) and
        # produces one feature vector per patch, summarizing what that tile
        # looks like.
        image_features = self.get_model().get_vision_tower()(images)
        #print('image_features.shape: ', image_features.shape)
        # [1, 576, 1024]
        # Meaning: 1 image, 576 patch-tokens (a 24x24 grid of patches, since
        # 24*24=576), each described by 1024 numbers.

        # Step 2: run those 1024-number vectors through the projector
        # ("translator" network) to convert them into 4096-number vectors,
        # which is the size the language model's own word-embeddings use.
        # Now these vision vectors are numerically compatible and can be
        # mixed directly into the sequence of word embeddings.
        image_features = self.get_model().mm_projector(image_features)
        #print('image_features.shape: ', image_features.shape)
        # [1, 576, 4096]
        #assert False
        return image_features

    def generate_scene_level_features_option_3(self, scene_point_feature_map):
        # PLAIN-LANGUAGE EXPLANATION OF THE WHOLE FUNCTION:
        # In this autonomous-driving project, instead of a camera image, we
        # often have a "bird's-eye-view" (BEV) feature map: imagine looking
        # straight down at the road from above, where every little square
        # of the ground has a vector of numbers describing what a sensor
        # (e.g. LiDAR/radar processed by a perception network) detected
        # there. This function chops that 2-D grid of ground-squares into
        # bigger rectangular "patches" (similar to how encode_images chops
        # an image into 24x24 patches), and for each patch, squashes all
        # the numbers inside it into one single long vector. This turns the
        # 2-D sensor map into a list of "tokens" -- just like image patches
        # become a list of image tokens -- so it can later be projected and
        # fed into the language model the same way image tokens would be.
        #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
        # [32, 2, 2, 256, 50, 88]
        # [batch_size, num_input_frames, num_cavs, feature_size, spatial_left_right, spatial_forward_backward]
        # Meaning of each dimension:
        #  - batch_size = 32: 32 training examples processed together at once.
        #  - num_input_frames = 2: 2 separate time steps/snapshots of the scene.
        #  - num_cavs = 2: 2 connected vehicles, each contributing its own view.
        #  - feature_size = 256: each ground-square has a 256-number description.
        #  - spatial_left_right = 50, spatial_forward_backward = 88: the
        #    bird's-eye-view grid is 50 squares in one direction and 88 in
        #    the other (like rows and columns of a big grid over the road).

        # pad spatial_left_right to 51
        # patch size [3, 4] to generate 17 * 22 = 374 tokens
        # inside each patch, linearize the feature
        # each token's feature size is 256 * 3 * 4 = 3072
        # then apply projector to have feature size 1024
        point_features = scene_point_feature_map
        # [32, 2, 2, 256, 50, 88]

        # Step 1, padding spatial_left_right to 51
        # PyTorch's `pad` function needs the grid dimensions to divide
        # evenly by the patch size we're about to use (3 in this
        # direction). 50 doesn't divide evenly by 3, but 51 does (51 = 3*17),
        # so we add one extra row of zeros to make the math work out. This
        # is just bookkeeping/padding, similar to adding a blank margin.
        p2d = (0, 0, 0, 1)
        # (0, 0, 0, 1) tells `pad` to add 0 extra columns before/after the
        # very last dimension, and 0 before / 1 after the second-to-last
        # dimension (i.e. one extra row of zeros at the end of the
        # spatial_left_right=50 dimension).
        point_features = nn.functional.pad(point_features, p2d, "constant", 0)
        #print('point_features.shape: ', point_features.shape)
        # [32, 2, 2, 256, 51, 88]

        # Step 2, patch [51, 88] to [3, 4] * [17, 22]
        # currently only 4-D tensors are supported
        # nn.functional.unfold is a PyTorch operation that slides a
        # rectangular window (here 3 rows x 4 columns) across a 2-D grid and
        # extracts each window's contents flattened into a single long
        # vector (this is the "patchify + linearize" step described in the
        # comment above). It ONLY works on 4-D tensors shaped
        # [batch, channels, height, width], so first we have to squash the
        # batch_size / num_input_frames / num_cavs dimensions together into
        # one big "batch" dimension.
        batch_size, num_input_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1 = point_features.shape
        point_features = point_features.reshape([batch_size * num_input_frames * num_cavs, feature_size, spatial_dim_0, spatial_dim_1])
        #print('point_features.shape: ', point_features.shape)
        # [32 * 2 * 2, 256, 51, 88]
        point_features = nn.functional.unfold(point_features, (3, 4), stride=(3,4))
        # (3, 4) is the patch window size (3 rows x 4 columns of the grid),
        # and stride=(3,4) (same as the window size) means non-overlapping
        # patches, i.e. we chop the grid into a checkerboard of tiles rather
        # than sliding with overlap.
        #print('point_features.shape: ', point_features.shape)
        # [32 * 2 * 2, 256 * 3 * 4, 17 * 22]
        # [32 * 2 * 2, 3072, 374]
        # 51 / 3 = 17 patches tall, 88 / 4 = 22 patches wide, so
        # 17*22 = 374 total patches ("tokens"). Each patch's contents (256
        # numbers per ground-square, times 3x4=12 ground-squares per patch)
        # get flattened into one 3072-number vector (256 * 3 * 4 = 3072).

        _, new_feature_size, num_tokens = point_features.shape
        point_features = point_features.reshape([batch_size, num_input_frames, num_cavs, new_feature_size, num_tokens])
        # Undo the earlier squashing, restoring the batch_size /
        # num_input_frames / num_cavs dimensions as separate axes again.
        #print('point_features.shape: ', point_features.shape)
        # [32, 2, 2, 3072, 374]

        return point_features


    def generate_scene_level_features(self, my_model_config, scene_point_feature_map, regression_map, classification_map, active_agent_mask):
        '''
        Input:
          regression_map
            [batch_size, num_input_frames=2, num_cavs=2, feature_size=14, spatial_dim_0=50, spatial_dim_1=88]
          classification_map
            [batch_size, num_input_frames=2, num_cavs=2, feature_size=2, spatial_dim_0=50, spatial_dim_1=88]
          active_agent_mask
            [batch_size, num_input_frames=2, num_cavs=4] bool indicate whether agent i is active
        Output:
          scene_level_features: list of scene_level_feature, list size=num_cavs
            scene_level_feature: [batch_size, num_tokens=num_patches=220, feature_size=4096]
        '''
        # PLAIN-LANGUAGE EXPLANATION:
        # This function builds "scene-level" tokens -- numeric summaries of
        # the whole driving scene as seen from a bird's-eye-view, per
        # vehicle and per time frame. There are two different ways
        # ("modes") this project has experimented with to build these
        # tokens, selected by my_model_config['scene_feature_mode']:
        #   'deep'    -> uses rich, high-dimensional internal features
        #                straight from the perception network
        #                (scene_point_feature_map), which tend to carry
        #                more information but are more expensive/complex.
        #   'shallow' -> uses only the perception network's final, simple
        #                OUTPUTS: "regression_map" (predicted box shape and
        #                position numbers for every ground-square) and
        #                "classification_ map" (predicted object-vs-not
        #                probabilities for every ground-square). This is a
        #                simpler, lower-dimensional representation.
        # Whichever mode is used, the end goal is the same: end up with a
        # tensor shaped [batch_size, num_input_frames, num_cavs, num_tokens,
        # feature_size=4096] -- a sequence of 4096-number "tokens" per
        # vehicle per frame, ready to be mixed into the language model's
        # input the same way image tokens are.

        # Scene-level

        if my_model_config['scene_feature_mode'] == 'deep':

          # New approach: deep features from scene_point_feature_map
          #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
          # [32, 2, 2, 256, 50, 88]
          # [32, 2, 1, 256, 48, 128] # cobevt
          scene_level_features = self.generate_scene_level_features_option_3(scene_point_feature_map)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 3072, 374]
          # [32, 2, 2, 256 * 3 * 4, 17 * 22]
          # [batch_size, num_input_frames, num_cavs, feature_size, num_tokens]
          # [32, 2, 2, 3072, 512] cobevt
          #assert False

          # swap axis
          # torch.swapaxes just switches the order of two dimensions
          # without changing any actual numbers -- here it swaps
          # "feature_size" and "num_tokens" so the shape matches the
          # standard "[..., num_tokens, feature_size]" layout used
          # everywhere else in this codebase (and expected by the
          # projector, similar to how image_features are shaped
          # [batch, num_patches, feature_size]).
          #scene_level_features = torch.permute(scene_level_features, (0, 1, 3, 2))
          scene_level_features = torch.swapaxes(scene_level_features, -1, -2)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 374, 3072] # new option 3
          # [batch_size, num_input_frames, num_cavs, num_tokens, feature_size]
          #assert False


        # [batch_size, feature_size, spatial_left_right, spatial_forward_backward]
        # patch size [2, 2] to generate 25 * 44 = 1100 tokens
        # inside each patch, linearize the feature
        # each token's feature size is 256 * 2 * 2 = 1024
        # then apply projector to have feature size 4096
        # (The following commented-out block below is an OLDER, abandoned
        # version of the 'deep' code path kept for reference/history; it is
        # not executed since it's all commented out.)

        ####point_features = scene_point_feature_map
        # Step 2, patch [50, 88] to [2, 2] * [25, 44]
        ####point_features = nn.functional.unfold(point_features, (2, 2), stride=(2, 2))
        #print('point_features.shape: ', point_features.shape)
        # [32, 512*2*2=2048, 25*44=1100]
        # [2, 256 * 2 * 2, 25 * 44]
        # [batch_size, feature_size, num_tokens]

        # Step 3, swap axis to have the same format of image_features:
        ####point_features = torch.permute(point_features, (0, 2, 1))
        #print('point_features.shape: ', point_features.shape)
        # [32, 1100, 2048]
        # [2, 1100, 1024]
        # [batch_size, num_point_tokens, feature_size]

        # Step 4, apply projector
        ####point_features = self.get_model().mm_projector(point_features)
        #print('point_features.shape: ', point_features.shape)
        # [2, 1100, 4096]
        # [batch_size, num_point_tokens, feature_size]


        elif my_model_config['scene_feature_mode'] == 'shallow':
          #print('my_model_config: ', my_model_config)
          # Old approach before 0926
          # Old approach: shallow features from reg and cls maps
          #print('regression_map.shape: ', regression_map.shape)
          # [32, 2, 14, 50, 88]
          # v2xreal [32, 1, 4, 42, 100, 176]
          #print('classification_map.shape: ', classification_map.shape)
          # [32, 2, 2, 50, 88]
          # v2xreal [32, 1, 4, 18, 100, 176]

          # split to separate map per cav
          # patch size [5, 4] to generate (50/5) * (88/4) = 220 tokens
          # feature size (14 + 2 = 16) * 5 * 4  =  320 for each cav
          # pad 320 to 1024 before applying projector
          # split to separate map per cav
          #regression_map = torch.chunk(regression_map, num_cavs, 1)
          #print('regression_map[0].shape: ', regression_map[0].shape)
          #classification_map = torch.chunk(classification_map, num_cavs, 1)
          #scene_level_features = [torch.cat([regression_map[i], classification_map[i]], dim=1) for i in range(num_cavs)]
          #print('scene_level_features[0].shape: ', scene_level_features[0].shape)
          #print('scene_level_features[1].shape: ', scene_level_features[1].shape)
          # [32, 16, 50, 88]

          # Glue the regression (box-shape/position predictions) and
          # classification (object-probability predictions) maps together
          # along the "feature_size" axis -- i.e. for every ground-square,
          # stack its regression numbers and its classification numbers
          # into one combined list of numbers.
          scene_level_features = torch.cat([regression_map, classification_map], dim=3)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 16, 50, 88]
          # [32, 2, 1, 16, 48, 128] # cobevt
          # v2xreal [32, 1, 4, 60, 100, 176]
          batch_size, num_input_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1 = scene_level_features.shape
          scene_level_features = scene_level_features.reshape([batch_size * num_input_frames * num_cavs, feature_size, spatial_dim_0, spatial_dim_1])
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32 * 2 * 2, 16, 50, 88]
          # v2xreal [128 = 32 * 1 * 4, 60 = 42 + 18, 100, 176]

          # For v2xreal, we need to reduce the spatial dimention
          if my_model_config['dataset_source'] == 'v2xreal':
            # avg_pool2d shrinks the grid by averaging together each 2x2
            # block of ground-squares into one square -- this is a common
            # way to reduce the resolution/size of a feature map while
            # keeping a smoothed-out summary of the values, used here
            # because the v2xreal dataset's maps are a different
            # (larger) size than v2v4real's and need to be shrunk down
            # to a comparable size before patchifying.
            scene_level_features = nn.functional.avg_pool2d(scene_level_features, kernel_size=2, stride=2)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # v2xreal [128, 60, 50, 88]


          # only 4-D tensor is supported
          # patch size [5, 4] to generate (50/5) * (88/4) = 220 tokens
          patch_size = (5, 4) # v2v4real
          if my_model_config['dataset_source'] == 'v2xreal':
              patch_size = (4, 4)

          #scene_level_features = nn.functional.unfold(scene_level_features,  (5, 4), stride=(5, 4))
          # Same "chop the grid into non-overlapping rectangular patches and
          # flatten each patch into one vector" idea explained in
          # generate_scene_level_features_option_3 above.
          scene_level_features = nn.functional.unfold(scene_level_features,  patch_size, stride=patch_size)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32 * 2 * 2, 16*5*4, 50/5 * 88/4]
          # v2xreal ([128=32*1*4, 960=60*4*4, 264=50//4 * 88/4]
          _, new_feature_size, num_tokens = scene_level_features.shape

          scene_level_features = scene_level_features.reshape([batch_size, num_input_frames, num_cavs, new_feature_size, num_tokens])
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 16*5*4, 50/5 * 88/4]
          # [32, 2, 2, 320, 220]
          # [32, 2, 1, 320, 288] cobevt [48, 128]
          # v2xreal [32, 1, 4, 960, 264]

          #assert False


          # swap axis
          #scene_level_features = torch.permute(scene_level_features, (0, 1, 3, 2))
          scene_level_features = torch.swapaxes(scene_level_features, -1, -2)

          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 220, 320] # old shallow option 1
          # [batch_size, num_input_frames, num_cavs, num_tokens, feature_size]
          # v2xreal [32, 1, 4, 264, 960]

          # pad 320 to 1024 before applying projector
          # pre-pad, different from object level append-pad
          # The regular mm_projector (used for images/objects too) was
          # built expecting input vectors of size 1024. The "shallow" scene
          # features here are only 320 numbers long, so we pad extra zeros
          # onto the FRONT of each vector to stretch it to length 1024
          # without changing any of the real numbers, just so it fits
          # through the same projector network. (Padding with zeros doesn't
          # add any new information -- it just makes the tensor the right
          # shape so the matrix multiplication inside the projector works.)
          mm_projector_input_size = 1024
          scene_level_features = nn.functional.pad(
              scene_level_features,
              (mm_projector_input_size - scene_level_features.shape[-1], 0),
              'constant',
              0
          )
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 220, 1024]
          # v2xreal [32, 1, 4, 264, 1024]
          #assert False

        else:
          print('not implemented')
          assert False


        # apply projector
        # MY_DEBUG
        # Finally, run the assembled scene-level vectors through whichever
        # projector network matches this mode: the shared mm_projector for
        # 'shallow' mode (since those got padded up to the same 1024 size
        # everything else uses), or the dedicated mm_scene_projector for
        # 'deep' mode (since those keep their own, different-sized 3072
        # input, so they need a projector built specifically to accept
        # that size).
        if my_model_config['scene_feature_mode'] == 'shallow':
          scene_level_features = self.get_model().mm_projector(scene_level_features)
        elif my_model_config['scene_feature_mode'] == 'deep':
          # New approach: do not do padding, directly set mm_scene_projector's input size to 3072
          scene_level_features = self.get_model().mm_scene_projector(scene_level_features)
        else:
          assert False

        #print('scene_level_features.shape: ', scene_level_features.shape)
        # [32, 2, 2, 220, 4096] # old shallow option 1
        # [32, 2, 2, 374, 4096] # new deep option 3
        #
        # cobevt
        # [32, 2, 1, 288, 4096] # old shallow option 1
        # [32, 2, 1, 512, 4096] # new deep option 3
        #
        # v2xreal
        # [32, 1, 4, 264, 4096]

        return scene_level_features


    def get_positional_embedding(self, detection_box_score, hidden_dim):
        '''
        Similar to
        https://github.com/eddyhkchiu/DMSTrack/blob/master/DMSTrack/model.py#L202

        Input:
          detection_box_score:
            [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, 7 + 1 parameters]
          hidden_dim:
            constant for positional coding

        Output:
          positional_embedding: [batch_size, max_num_boxes_per_cav * num_cavs, feature_size=8*hidden_dim]
        '''
        # PLAIN-LANGUAGE EXPLANATION:
        # Neural networks like transformers process a "bag" of numbers and
        # have no built-in sense of the actual real-world meaning of "3.5
        # meters" vs "3.6 meters" -- these raw numbers are hard for the
        # network to use directly for fine distinctions. A common trick
        # (borrowed from the original Transformer paper's "positional
        # encoding", and reused here for encoding a detected object's
        # location/size/etc instead of its position-in-a-sentence) is to
        # convert each raw number into a whole pattern of sine and cosine
        # wave values at many different frequencies. This gives the network
        # a much richer, smoother way to represent "how big" or "how far
        # away" something is, making small differences easier to learn
        # from, in the same way that a musical chord (many pure tones
        # combined) can encode more nuance than a single flat tone.
        positional_embedding = None
        #print('detection_box_score.shape: ', detection_box_score.shape)
        # [32, 2, 2, 50, 8]
        # 8: [h, w, l, x, y, z, a, s]
        # meaning: height, width, length (the 3-D size of a detected
        # object's bounding box), x/y/z (its position in 3-D space), a
        # (its orientation angle), s (a confidence score).
        batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, num_parameters = detection_box_score.shape

        #print('hidden_dim: ', hidden_dim)
        # 128

        positional_feature = torch.reshape(detection_box_score, [batch_size * num_input_frames * num_cavs *  max_num_boxes_per_cav, num_parameters])
        #print('positional_feature.shape: ', positional_feature.shape)

        # normalize all distance by dividing by max distance 200 meters,
        # before applying sin cos positional embedding
        # TODO: move this constant to dataset dependent config
        max_distance = 200
        # Divide the first 6 numbers (h, w, l, x, y, z -- all distances,
        # measured in meters) by 200 so their values end up roughly between
        # -1 and 1. Neural networks generally train better/faster when
        # their input numbers are kept in a small, consistent range rather
        # than spanning huge or wildly different scales.
        positional_feature[:, :6] /= max_distance
        #print('positional_feature: ', positional_feature)
        # after this normalization, the range of distances is [-1, 1]

        half_hidden_dim = hidden_dim // 2
        # because the range of distance is [-1, 1]
        # we want scale = math.pi
        # the original code scale = 2 * math.pi, is for range [0, 1]
        scale = math.pi

        # Build a set of "frequencies" -- essentially a list of numbers
        # that will stretch or compress the sine/cosine waves used below,
        # so that different positions in the final "hidden_dim"-long vector
        # capture patterns at different scales (some very fine-grained,
        # some very coarse), similar to how a musical chord layers many
        # different pitches together.
        dim_t = torch.arange(half_hidden_dim, dtype=positional_feature.dtype, device=positional_feature.device)
        dim_t = 2 ** (2 * dim_t / hidden_dim)
        #print('dim_t[:5]: ', dim_t[:5])
        # (64)

        positional_embedding = positional_feature.unsqueeze(dim=2)
        # unsqueeze adds a new dimension of size 1 -- here it turns each of
        # the 8 raw numbers per box into a 1-number "starting point" so it
        # can be multiplied against all 64 frequency values below (this is
        # a standard trick called "broadcasting": PyTorch automatically
        # repeats values across a size-1 dimension so shapes line up for
        # multiplication).
        #print('positional_embedding.shape: ', positional_embedding.shape)
        # [3200, 8, 1]
        # (batch_size * max_num_boxes, num_parameters, 1)

        positional_embedding = positional_embedding * scale / dim_t
        #print('positional_embedding.shape: ', positional_embedding.shape)
        # [3200, 8, 64]
        # (batch_size * max_num_boxes, num_parameters, half_hidden_dim)
        #print('positional_embedding[0, -1, :5]: ', positional_embedding[0, -1, :5])

        # Apply the sine function to half of the values and cosine to the
        # other half, then glue them back together. This sin/cos pairing is
        # the classic "positional encoding" trick -- it lets the model
        # later recover both the value and, through combinations of
        # multiple frequencies, fine relative differences between values.
        positional_embedding = torch.cat([positional_embedding.sin(), positional_embedding.cos()], dim=2)
        #print('positional_embedding.shape: ', positional_embedding.shape)
        # [3200, 8, 128]
        # (batch_size * max_num_boxes, num_parameters, hidden_dim)
        # print encoded x
        #print('positional_embedding[:, 3, :]: ', positional_embedding[:, 3, :])

        # final reshape
        # Flatten the "8 parameters x 128 hidden_dim" pair for each box
        # into a single 1024-number vector (8*128=1024) per box, and
        # restore the batch_size / num_input_frames / num_cavs /
        # max_num_boxes_per_cav dimensions.
        positional_embedding = positional_embedding.reshape([batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, num_parameters * hidden_dim])
        #print('positional_embedding.shape: ', positional_embedding.shape)
        # [32, 2, 2, 50, 8*128=1024]

        #assert False
        return positional_embedding


    def generate_object_level_features(self, my_model_config, detection_box_score, object_features):
        '''
        Input:
          detection_box_score:
            [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, 7 + 1 parameters]
          object_features:
            [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, 256 feature values]

        Output:
          object_level_feature: [batch_size, num_input_frames, num_cavs, num_tokens=max_num_objects=50, feature_size=4096]
        '''
        # PLAIN-LANGUAGE EXPLANATION:
        # While generate_scene_level_features summarizes the WHOLE
        # bird's-eye-view grid, this function instead builds one token PER
        # DETECTED OBJECT (e.g. one token per car/pedestrian the perception
        # system found), using that object's predicted box (position, size,
        # orientation, confidence) and/or its internal neural-network
        # feature vector. Up to `max_num_boxes_per_cav` objects are kept per
        # vehicle per frame (extra "slots" beyond the real detected objects
        # are presumably filled with zeros/padding elsewhere in the data
        # pipeline). Just like generate_scene_level_features, there are
        # multiple modes for exactly what information goes into each
        # object's token.
        #print('object_features.shape: ', object_features.shape)
        # [32, 2, 2, 50, 256]
        # [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, feature_size]
        batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, feature_size = object_features.shape


        if my_model_config['object_feature_mode'] == 'deep':
          # New approach:
          # concat detection_box_score and object_features
          # Glue each object's 8-number box description (h, w, l, x, y, z,
          # angle, score) together with its 256-number internal feature
          # vector, forming one longer combined vector per object.
          object_level_features = torch.cat([
            detection_box_score,
            object_features
          ], dim=-1)
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, 2, 2, 50, 264]
          #assert False

          # Pad with zeros up to size 1024 (same reasoning as in the
          # 'shallow' scene-feature branch above) so it fits the shared
          # projector's expected input size. Here the padding is added at
          # the END of the vector instead of the front (compare with the
          # scene-level 'shallow' padding above, which pads at the front) --
          # this doesn't matter mathematically, just a stylistic choice by
          # whoever wrote the two code paths.
          mm_projector_input_size = 1024
          object_level_features = nn.functional.pad(
              object_level_features,
              (0, mm_projector_input_size - object_level_features.shape[-1]),
              'constant',
              0
          )
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, 2, 2, 50, 1024]
          #assert False


        elif my_model_config['object_feature_mode'] == 'shallow':
          # Old approach 0928
          # detection_box_score only

          # Object-level
          # detection_box_score
          # list of batch_sample
          # each has (num_boxes, box_score_feature_size=8)
          # box_score_feature: [h, w, l, x, y, z, a, s]
          #print('detection_box_score.shape: ', detection_box_score.shape)
          # [32, 2, 2, 50, 8] [batch_size, num_cavs, max_num_boxes_per_cav, 7 + 1 parameters]
          # pad zero to feature size mm_projector_input_size (original 1024)
          # In "shallow" mode we use ONLY the raw 8-number box description
          # (no internal 256-number feature vector at all), just padded out
          # to size 1024 with zeros so it can pass through the same
          # projector.
          mm_projector_input_size = 1024
          detection_box_score = nn.functional.pad(
              detection_box_score,
              (0, mm_projector_input_size - detection_box_score.shape[-1]),
              'constant',
              0
          )
          #print('detection_box_score.shape: ', detection_box_score.shape)
          # [32, 2, 2, 50, 1024]
          object_level_features = detection_box_score

          # TODO: better way is to use different projector to make feature size from 7 to 4096
          #object_level_features = torch.zeros([batch_size, 50, 1024], dtype=detection_box_score[0].dtype, device=detection_box_score[0].device)
          #for i in range(batch_size):
          #    object_level_features[i, :detection_box_score[i].shape[0], :detection_box_score[i].shape[1]] = detection_box_score[i]
          #print('object_level_features.shape: ', object_level_features.shape)
          #print('object_level_features[0, :10, :10]: ', object_level_features[0, :10, :10])
          #print('object_level_features[1, :10, :10]: ', object_level_features[1, :10, :10])

        elif my_model_config['object_feature_mode'] == 'pos128':
          # A third mode: instead of raw numbers (possibly padded) or a
          # concatenation of raw + deep features, use the rich sine/cosine
          # "positional embedding" trick (explained in detail inside
          # get_positional_embedding above) to encode each object's box
          # description into a richer 1024-number vector (8 parameters *
          # 128 hidden_dim = 1024).
          #print('detection_box_score.shape: ', detection_box_score.shape)
          # [32, 2, 50, 8] [batch_size, num_cavs, max_num_boxes_per_cav, 7 + 1 parameters]
          hidden_dim = 128
          positional_embedding = self.get_positional_embedding(detection_box_score, hidden_dim)
          positional_feature_size = positional_embedding.shape[-1]

          object_level_features = positional_embedding
          #print('object_level_features.shape: ', object_level_features.shape)
          # [batch_size=32, num_input_frames=2, num_cavs=2, max_num_boxes_per_cav=50, 8*128=1024]

        else:
          print('not implemented')
          assert False


        # TODO: use different projector
        # Whatever mode was used above, the final 1024-number-per-object
        # vectors now get passed through the SAME shared mm_projector used
        # for images and 'shallow'-mode scene features, converting them to
        # 4096-number vectors that match the language model's embedding
        # size.
        object_level_features = self.get_model().mm_projector(object_level_features)
        #print('object_level_features.shape: ', object_level_features.shape)
        # [32, 2, 2, 50, 4096]
        # [batch_size=32, num_input_frames=2, num_cavs=2, max_num_boxes_per_cav=50, llm_feature_size=4096]
        # v2vreal [32, 1, 4, 100, 4096]
        #assert False

        return object_level_features


    def concat_features_original(self, scene_level_features, object_level_features):
        '''
        Original approach:
          [num_input_frames, cav_id, scene_or_object]
          f_0
            cav_ego_scene, cav_ego_object, cav_1_scene, cav_1_object
          f_1
            cav_ego_scene, cav_ego_object, cav_1_scene, cav_1_object

        Input:
          scene_level_features: [batch_size, num_input_frames, num_cavs, num_tokens, feature_size=4096]
          object_level_features: [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, feature_size=4096]
        Output:
          point_features: [batch_size, final_num_tokens, feature_size=4096]
        '''
        # PLAIN-LANGUAGE EXPLANATION:
        # We now have, for every (time frame, vehicle) pair, both a chunk of
        # "scene-level" tokens (the whole bird's-eye-view summary) and a
        # chunk of "object-level" tokens (one per detected object). This
        # function stitches ALL of these token chunks together, one after
        # another, into a single long sequence -- ordered as: for frame 0:
        # vehicle 0's scene tokens, then vehicle 0's object tokens, then
        # vehicle 1's scene tokens, then vehicle 1's object tokens, and so
        # on for however many vehicles (num_cavs) there are; then repeat the
        # same pattern for frame 1, etc. This ordering is exactly what the
        # docstring's small diagram above is illustrating. The final result
        # is one big sequence of "tokens" per training example -- exactly
        # analogous to how encode_images ends up with a sequence of image
        # patch tokens -- ready to be spliced into the language model's
        # input in place of the "<image>" placeholder in the prompt.
        batch_size, num_input_frames, num_cavs, _, feature_size = scene_level_features.shape

        point_features_all_frames = []
        for f in range(num_input_frames):
            # torch.cat glues tensors together end-to-end along a chosen
            # dimension (dim=1 here means "along the token dimension", so
            # we're literally lengthening the sequence of tokens, not
            # changing feature_size or batch_size). The specific
            # if/elif/else below just handles different possible numbers of
            # connected vehicles (1, 2, 3, or 4) since the exact list of
            # tensors to concatenate depends on how many vehicles are
            # present.

          if num_cavs == 4:
            single_frame_point_features = torch.cat([
              scene_level_features[:, f, 0, :, :],
              object_level_features[:, f, 0, :, :],
              scene_level_features[:, f, 1, :, :],
              object_level_features[:, f, 1, :, :],
              scene_level_features[:, f, 2, :, :],
              object_level_features[:, f, 2, :, :],
              scene_level_features[:, f, 3, :, :],
              object_level_features[:, f, 3, :, :],
            ],  dim=1)
          elif num_cavs == 3:
            single_frame_point_features = torch.cat([
              scene_level_features[:, f, 0, :, :],
              object_level_features[:, f, 0, :, :],
              scene_level_features[:, f, 1, :, :],
              object_level_features[:, f, 1, :, :],
              scene_level_features[:, f, 2, :, :],
              object_level_features[:, f, 2, :, :],
            ],  dim=1)
          elif num_cavs == 2:
            single_frame_point_features = torch.cat([
              scene_level_features[:, f, 0, :, :],
              object_level_features[:, f, 0, :, :],
              scene_level_features[:, f, 1, :, :],
              object_level_features[:, f, 1, :, :],
            ],  dim=1)
          else:
            assert(num_cavs == 1)
            single_frame_point_features = torch.cat([
              scene_level_features[:, f, 0, :, :],
              object_level_features[:, f, 0, :, :],
            ],  dim=1)

          point_features_all_frames.append(single_frame_point_features)

        # Finally, glue the (already-combined-per-frame) token sequences
        # from ALL time frames together too, end to end -- giving one huge
        # sequence of tokens covering every frame and every vehicle.
        point_features = torch.cat(point_features_all_frames, dim=1)
        #print('point_features.shape: ', point_features.shape)
        # num_cavs == 2
        # [32, 540*num_input_frames , 4096] # shallow
        # [32, 848*num_input_frames , 4096] # deep
        # num_cavs == 1
        # [32, 270*num_input_frames , 4096] # shallow
        # [32, 424*num_input_frames , 4096] # deep
        #
        # v2xreal
        # num_cavs == 4
        # [32, 1456*num_input_frames, 4096]
        #assert False
        return point_features


    def generate_point_features(self, my_model_config, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask):
        # PLAIN-LANGUAGE EXPLANATION:
        # This is the top-level function that decides WHICH kind(s) of
        # sensor-derived tokens to build for a given training example,
        # based on config flags, and returns the final combined sequence
        # of tokens (still called "point_features" here, as a holdover
        # from when this project only used raw LiDAR point clouds).
        #print("my_model_config['ego_only']: ", my_model_config['ego_only'])
        if my_model_config['ego_only']:
          cav_ids = ['ego']
          # "ego" = the vehicle we personally care about / are asking the
          # language model to reason from the perspective of (a common term
          # in self-driving research meaning "our own vehicle", as opposed
          # to other, nearby vehicles).
        else:
          cav_ids = ['ego', '1']
        #print('cav_ids: ', cav_ids)
        # NOTE: cav_ids computed above doesn't actually appear to be used
        # further down in this function -- looks like leftover/unused code
        # from an earlier version.

        scene_level_only = my_model_config['scene_level_only']
        object_level_only = my_model_config['object_level_only']
        if scene_level_only:
          # Only build scene-level (whole bird's-eye-view) tokens, skip
          # object-level tokens entirely.
          scene_level_features = self.generate_scene_level_features(my_model_config, scene_point_feature_map, regression_map, classification_map, active_agent_mask)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [batch_size, num_input_frames, num_cavs, num_tokens, feature_size=4096]
          # [32, 2, 2, 220, 4096] shallow
          # [32, 2, 2, 374, 4096] deep
          batch_size, num_input_frames, num_cavs, num_tokens, feature_size = scene_level_features.shape
          # Flatten num_input_frames, num_cavs, and num_tokens together
          # into one single "sequence length" dimension, since ultimately
          # the language model just wants one flat sequence of tokens per
          # training example, not a multi-dimensional grid of them.
          scene_level_features = scene_level_features.reshape([batch_size, num_input_frames * num_cavs * num_tokens, feature_size])
          #print('scene_level_features.shape: ', scene_level_features.shape)
          #assert False
          return scene_level_features
        elif object_level_only:
          # MY_DEBUG
          # only use object-level features
          # Only build object-level (per-detected-object) tokens, skip
          # scene-level tokens entirely.
          object_level_features = self.generate_object_level_features(my_model_config, detection_box_score, object_features)
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, num_input_frames=2, num_cavs=2, num_tokens=50, feature_size=4096]
          batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, feature_size = object_level_features.shape
          object_level_features = object_level_features.reshape([batch_size, num_input_frames * num_cavs * max_num_boxes_per_cav, feature_size])
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, 2*2*50, 4096]
          #assert False
          return object_level_features
        else: # both scene and object
          # Default/full case: build BOTH scene-level and object-level
          # tokens, and interleave them together (per vehicle, per frame)
          # using concat_features_original, described in detail above.
          scene_level_features = self.generate_scene_level_features(my_model_config, scene_point_feature_map, regression_map, classification_map, active_agent_mask)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [batch_size, num_input_frames, num_cavs, num_tokens, feature_size=4096]
          # [32, 2, 2, 220, 4096] shallow
          # [32, 2, 2, 374, 4096] deep
          #
          # cobevt
          # [32, 2, 1, 288, 4096] # old shallow option 1
          # [32, 2, 1, 512, 4096] # new deep option 3
          #
          # v2xreal
          # [32, 1, 4, 264, 4096]
          #assert False

          object_level_features = self.generate_object_level_features(my_model_config, detection_box_score, object_features)
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, num_input_frames=2, num_cavs=2, max_num_boxes_per_cav=50, feature_size=4096]
          # v2xreal [32, 1, 4, 100, 4096])

          # Concat scene-level tokens and object-level tokens for each cav
          point_features = self.concat_features_original(scene_level_features, object_level_features)

          #print('point_features.shape: ', point_features.shape)
          # no fusion, two cavs
          # [32, 540 * num_input_frames , 4096] # shallow
          # [32, 848 * num_input_frames , 4096] # deep
          # [batch_size, final_num_tokens, feature_size]
          #
          # cobevt, one merged
          # [32, 338 * num_input_frames , 4096] # shallow
          # [32, 562 * num_input_frames , 4096] # deep
          # [batch_size, final_num_tokens, feature_size]
          #
          # v2xreal [32, 1456 * num_input_frames, 4096]

          return point_features


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, my_model_config=None, scene_point_feature_map=None, regression_map=None, classification_map=None, detection_box_score=None, object_features=None, active_agent_mask=None
    ):
        # PLAIN-LANGUAGE EXPLANATION OF THE WHOLE FUNCTION:
        # This is the central "glue" function that runs right before the
        # language model itself processes a batch of examples. It takes the
        # tokenized text (input_ids -- a sequence of integer IDs, each
        # representing one word/sub-word, PLUS special placeholder IDs like
        # IMAGE_TOKEN_INDEX wherever an image/sensor-data should be
        # inserted) and:
        #   1. Converts the non-text data (images, or in this project's
        #      case, sensor feature maps) into a sequence of numeric
        #      "tokens" the same size as word embeddings (via the functions
        #      above).
        #   2. Converts the plain text tokens into their normal word
        #      embeddings using the language model's own embedding table
        #      (a big lookup table where every possible word/sub-word has
        #      a learned vector of numbers representing its meaning).
        #   3. Wherever the special image/sensor placeholder token appears
        #      in the text, CUTS the text embedding sequence there and
        #      SPLICES IN the full sequence of image/sensor tokens, so the
        #      final sequence looks like: [some words] [inserted
        #      image/sensor tokens] [some more words].
        #   4. Pads everything in the batch to the same total length (since
        #      different training examples end up with different total
        #      sequence lengths once the image/sensor tokens are spliced
        #      in), so they can be processed together efficiently as one
        #      rectangular batch tensor.
        # The very end result is a batch of ready-to-use numeric embedding
        # sequences (new_input_embeds) that get fed directly into the
        # language model's transformer layers, instead of the model doing
        # its own separate word-embedding lookup step.

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # If there's no vision tower configured, no image was actually
            # given, or we're in the special case of generating text
            # one-token-at-a-time during inference (input_ids.shape[1] == 1
            # means "we're only being asked about a single new token", which
            # happens during autoregressive generation after the first
            # step, when there's nothing new/multimodal to insert) --
            # just skip all the multimodal logic and return the inputs
            # unchanged, exactly as they came in.
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            # ------------------------------------------------------------
            # This whole branch is the ORIGINAL LLaVA code path for
            # handling actual camera images (including the "any
            # resolution"/multi-patch-per-image feature). It is NOT
            # exercised by this project's autonomous-driving experiments
            # (see the `assert False` a bit further down, which proves this
            # project always goes through the `else` branch instead, using
            # sensor data). It's kept here for reference/compatibility with
            # the original LLaVA codebase.
            # ------------------------------------------------------------
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # MY_CODE
            # for v2v4real experiment
            # use point cloud feature map and new projector to generate
            # point cloud feature tokens with shape [batch_size, num_point_feature_tokens, 4096]
            # make sure the new projector is trainable and inside the model checkpoint
            # TODO: use config arg to determine whether in llava image code path or v2v4real point code path
            #
            # THIS is the branch this project actually uses. It checks
            # whether any sensor-derived data was passed in (a scene
            # feature map or per-object detection scores) and, if so, calls
            # generate_point_features (defined above) to build the full
            # sequence of sensor-derived tokens.
            if scene_point_feature_map is not None or detection_box_score is not None:
                #print('my_model_config: ', my_model_config)
                #assert False
                #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
                point_features = self.generate_point_features(my_model_config, scene_point_feature_map, regression_map, classification_map,  detection_box_score, object_features, active_agent_mask)
                # and still call it image_features for now,
                # so that we do not need to change the remaining code in this function
                # (The variable is renamed "image_features" purely so the
                # rest of this function -- copied largely unmodified from
                # the original LLaVA project -- doesn't need to be rewritten
                # to use a different variable name; the underlying meaning
                # is "sensor-derived tokens", not real camera image tokens.)
                image_features = point_features
                #print('point_features.shape: ', point_features.shape)
                # [2, 374, 4096]
                # [16, 1150, 4096]
                # [16, 50 , 4096]
                # [batch_size, num_tokens, feature_size]
            else: # regular llava code path using image
                #print('simple image encoder code') # here
                image_features = self.encode_images(images)
                #print('image_features.shape: ', image_features.shape)
                # [1, 576, 4096]
                # This assert is just to check whether we accidentally comes to llava image code path
                # This assert deliberately crashes the program if this
                # branch is ever reached, as a safety-check/reminder that
                # this project is not expected to use plain camera images
                # through this code path -- if it ever does, something
                # unexpected is happening and it should fail loudly rather
                # than silently doing the wrong thing.
                assert False


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        # (Comment from the original LLaVA authors.) Below, several
        # optional inputs (labels, position_ids, attention_mask) are given
        # sensible default values if the caller didn't provide them, purely
        # to simplify the rest of the code so it doesn't need constant
        # "if this is None" checks everywhere.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            # An "attention mask" tells the model which positions in the
            # input are REAL content versus which are just padding filler
            # (so the model knows to ignore the padding). If none was
            # given, assume every position is real (all True/1s).
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            # "position_ids" tell the model the position (0th, 1st, 2nd,
            # ...) of each token in the sequence, which transformer models
            # need in order to understand word order (since, unlike RNNs,
            # transformers otherwise have no inherent sense of sequence
            # order). If none given, just number them 0, 1, 2, 3, ...
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            # "labels" are the correct next-word answers used to compute
            # the training loss. If none given (e.g. during pure
            # inference/generation, not training), fill with IGNORE_INDEX
            # everywhere, meaning "there's no loss to compute here".
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        #print('input_ids: ', input_ids)
        #print('labels: ', labels)
        #print('attention_mask: ', attention_mask)
        # remove the padding using attention_mask -- FIXME
        # Use the attention_mask to strip out padding positions from each
        # example, turning the neat rectangular batch tensor into a
        # (ragged/uneven-length) Python list of tensors, one per example,
        # each containing only its real (non-padding) tokens. This makes
        # the per-example splicing logic below much simpler, at the cost of
        # needing to re-pad everything back into a rectangle at the very
        # end of this function.
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        #print('input_ids[0][:10]: ', input_ids[0][:10])
        # [tensor([    1,  -200,   447,   688,  3391,   373,   263,   521,  8233,  4315,
        # 10348,  2909,    13], device='cuda:0')]

        #print('labels[0][:10]: ', labels[0][:10])
        # [tensor([ -100,  -100,   447,   688,  3391,   373,   263,   521,  8233,  4315,
        # 10348,  2909,    13], device='cuda:0')]
        #assert False

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # Loop through every example in the batch one at a time, since each
        # one may have a different number/placement of image/sensor
        # placeholder tokens and thus needs individual handling before
        # everything gets stacked back into a batch at the end.
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # Count how many "<image>" placeholder tokens are in this
            # particular example's text.
            if num_images == 0:
                # MY_DEBUG
                # TODO: trace the text-only code path
                # This example has NO image/sensor placeholder at all --
                # it's pure text. We still look up its normal word
                # embeddings, but to keep the code generic/simple we
                # concatenate an "empty slice" of the image features
                # (cur_image_features[0:0] -- taking zero rows from it)
                # which effectively adds nothing, just so the same
                # code shape works whether or not there were images.
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # embed_tokens is the language model's built-in lookup
                # table: give it a sequence of integer token IDs, get back
                # a sequence of learned numeric vectors (word embeddings),
                # one per token.
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # Find the exact positions (indices) in the token sequence
            # where the "<image>" placeholder(s) occur, and bookend that
            # list with -1 (representing "just before the very start") and
            # the sequence length (representing "just after the very end").
            # This turns the list of placeholder positions into a set of
            # boundary markers we can use to slice out all the "real text"
            # chunks that fall BETWEEN placeholders.
            #print('image_token_indices: ', image_token_indices)
            # [-1, 1, 13]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                # Slice out each chunk of real text tokens that sits
                # between two consecutive placeholder-boundary markers
                # (i.e. "noim" = "no image" -- just the plain text parts,
                # with the placeholder tokens themselves excluded/removed).
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            #print('cur_input_ids_noim: ', cur_input_ids_noim)
            # [tensor([1], device='cuda:0'), tensor([  447,   688,  3391,   373,   263,   521,  8233,  4315, 10348,  2909,
            # 13], device='cuda:0')]
            #print('cur_labels_noim: ', cur_labels_noim)
            #  [tensor([-100], device='cuda:0'), tensor([  447,   688,  3391,   373,   263,   521,  8233,  4315, 10348,  2909,
            # 13], device='cuda:0')]
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            #print('split_sizes: ', split_sizes)
            # [1, 11]

            # Look up embeddings for ALL the plain-text chunks at once (by
            # gluing them together first, doing one embedding lookup, then
            # splitting the result back apart) -- this is just an
            # efficiency trick; doing one big lookup is faster than many
            # small ones.
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            #print('cur_input_embeds.shape: ', cur_input_embeds.shape)
            # [12, 4096]
            #print('len(cur_input_embeds_no_im): ', len(cur_input_embeds_no_im))
            #print('cur_input_embeds_no_im[0].shape: ', cur_input_embeds_no_im[0].shape)
            # [1, 4096]
            #print('cur_input_embeds_no_im[1].shape: ', cur_input_embeds_no_im[1].shape)
            # [11, 4096]
            cur_new_input_embeds = []
            cur_new_labels = []

            # Now weave the plain-text chunks and the image/sensor token
            # chunks back together in the correct order: text chunk, then
            # image/sensor tokens, then next text chunk, then next
            # image/sensor tokens (if there were multiple placeholders),
            # and so on -- effectively "un-cutting" the sequence but with
            # real vision/sensor tokens now sitting where the placeholder
            # used to be.
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                #print('cur_new_input_embeds[i].shape: ', cur_new_input_embeds[i].shape)
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    #print('cur_image_features.shape: ', cur_image_features.shape)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    #print('cur_new_input_embeds[-1].shape: ', cur_new_input_embeds[-1].shape)
                    # For every inserted image/sensor token, its "label"
                    # (the correct next-word target used for computing
                    # training loss) is set to IGNORE_INDEX, because there
                    # is no "correct word" the model should have predicted
                    # at these positions -- they're not real words the
                    # model needs to learn to produce, they're data being
                    # fed IN, so the loss function should simply skip over
                    # them.
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            # Make sure every piece lives on the same device (e.g. the same
            # GPU) before gluing them together, since PyTorch requires all
            # tensors in an operation to be on the same device.

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            #print('cur_new_input_embeds.shape: ', cur_new_input_embeds.shape)
            # [1 + 576 + 11, 4096] == [588, 4096]
            cur_new_labels = torch.cat(cur_new_labels)
            #print('cur_new_labels.shape: ', cur_new_labels.shape)
            # [588]

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # Since inserting hundreds of image/sensor tokens can make some
        # examples' sequences much longer than the model's maximum
        # supported length, cut off anything beyond that maximum (this
        # simply discards the extra tokens/labels past the cutoff point).
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        # Now that every example in the batch potentially has a DIFFERENT
        # total sequence length (because different examples might have had
        # different numbers of placeholder tokens, or different lengths of
        # surrounding text), we need to pad them all out to the SAME length
        # so they can be stacked into one rectangular batch tensor again --
        # this is standard practice for batch processing in deep learning
        # frameworks, which require uniformly-shaped tensors.
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                #print('left')
                # "Left padding" means adding the filler/padding at the
                # BEGINNING of the sequence, so the real content is
                # right-aligned. Some models/tokenizer setups prefer this
                # (e.g. it can make generation slightly simpler).
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                #print('right') # here
                # "Right padding" (the default, and what's actually used in
                # this project, per the "here" debug print) means adding
                # the filler/padding at the END of the sequence instead, so
                # the real content is left-aligned.
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # torch.stack glues the individually-padded, now-equal-length
        # tensors back together into one single rectangular batch tensor
        # (adding a new "batch" dimension at the front), ready to be fed
        # straight into the language model.

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        #print('position_ids: ', position_ids) # None
        #print('attention_mask: ', attention_mask) # [1, 588]
        #print('past_key_values: ', past_key_values) # None
        #print('new_input_embeds.shape: ', new_input_embeds.shape) # [1, 588, 4096]
        #print('new_labels.shape: ', new_labels.shape) # [1, 588]
        #assert False
        # input_ids is returned as None here on purpose: from this point
        # onward, the model will be given ready-made numeric embeddings
        # (new_input_embeds) directly instead of raw token IDs, since the
        # image/sensor tokens spliced in have no corresponding "word ID" --
        # they were never looked up from the vocabulary table, they were
        # computed by the vision tower / projector networks instead.
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # MY_CODE
        # TODO: trace this function and see
        # if we need a similar one for point cloud feature
        #
        # PLAIN-LANGUAGE EXPLANATION:
        # This function optionally adds new special vocabulary
        # tokens/words (like literal text strings "<im_patch>", "<im_start>",
        # "<im_end>") to the tokenizer AND to the model's embedding table,
        # for setups that mark image regions with explicit start/end
        # text markers rather than (or in addition to) the single
        # IMAGE_TOKEN_INDEX placeholder trick used elsewhere in this file.
        # Adding new tokens to a tokenizer means the model's embedding
        # table (and its final output/prediction layer) need to be resized
        # to have room for these new entries.

        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            # resize_token_embeddings grows (or shrinks) the model's
            # embedding lookup table so it has exactly one row per token in
            # the tokenizer's vocabulary -- necessary any time new tokens
            # get added, since otherwise there'd be no learned vector for
            # the new token IDs to look up.

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data
                # "input embeddings" = the lookup table converting token
                # IDs into vectors (used at the start of the model).
                # "output embeddings" = the (often separate) table/weights
                # used at the very end of the model to convert its final
                # internal vector back into a probability distribution over
                # every possible next word.

                # Rather than initializing the brand-new tokens' vectors to
                # completely random numbers (which can destabilize
                # training, since these tokens start out totally
                # meaningless to the model), initialize them instead to the
                # AVERAGE of all the pre-existing tokens' vectors -- a
                # sensible, "neutral" starting point that's already roughly
                # in the same numeric range/style as real, trained word
                # embeddings.
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                # If we're in the training stage that ONLY trains the
                # projector ("mm_mlp_adapter") and keeps everything else
                # frozen, still allow the new special-token embeddings to
                # be trained (since they were just added and have no
                # meaningful values yet), but keep the output/prediction
                # layer frozen (since we don't want this early stage to
                # change how the model predicts words in general).
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                # MY_CODE
                # currently not hit
                assert False
                # Dead/unused code path (again guarded by an intentional
                # crash) for loading a separately pretrained embedding
                # table for these special tokens from an existing
                # checkpoint file, instead of using the "average of
                # existing tokens" initialization above.
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                # If we only added the image-patch token (not the
                # start/end tokens) and we're in the projector-only
                # training stage, keep BOTH the input and output embedding
                # tables entirely frozen -- this branch doesn't train any
                # new token embeddings at all.
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
