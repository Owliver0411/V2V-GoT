#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# =============================================================================
# BIG PICTURE / WHAT THIS FILE DOES
# =============================================================================
# This file's whole job is: given a path/name pointing at a saved model on
# disk (or on the Hugging Face Hub), figure out exactly WHICH kind of model
# it is (plain language model? LLaVA model? A LoRA fine-tune that needs to be
# combined with a separate base model? Etc.) and load it correctly into
# memory, ready to use.
#
# This is the "model builder" -- NOT the same file as the one that builds the
# vision tower/projector networks (that's a differently-named builder.py
# under multimodal_encoder/ and multimodal_projector/). This particular
# builder.py operates one level up: it loads the WHOLE model (language model
# + vision tower + projectors all together), typically for running inference
# (asking the model questions) rather than for training.
#
# A few AI/ML concepts that come up throughout this file, explained once
# here so the comments below can stay short:
#
#   - "Checkpoint": a saved snapshot of a model's numbers (weights) on disk,
#     usually as one or more files.
#
#   - "Tokenizer": a helper object that converts human-readable text into a
#     sequence of integer IDs the model understands (and back again). Every
#     language model is paired with a specific tokenizer trained alongside
#     it.
#
#   - "Base model" vs "delta"/"LoRA weights": Instead of publishing an
#     entire multi-gigabyte fine-tuned model, some setups only publish the
#     small set of CHANGES made on top of an existing, publicly available
#     base model. This saves bandwidth/storage and can also satisfy
#     licensing requirements (e.g. if the base model's license doesn't allow
#     redistributing modified full copies). To actually use the fine-tuned
#     model, you need to (a) download/load the original base model, and (b)
#     apply the saved changes on top of it. This file's `lora` code paths
#     do exactly that.
#
#   - "LoRA" (Low-Rank Adaptation): a popular, memory-efficient fine-tuning
#     technique. Instead of updating every single number in a giant
#     pretrained model (which is slow and memory-hungry), LoRA freezes the
#     entire original model and only trains a small number of extra,
#     "add-on" matrices bolted onto certain layers. During/after training,
#     these small add-on matrices can either be kept separate (so you can
#     easily switch them on/off, or swap in different LoRA fine-tunes on the
#     same base model) or "merged" directly into the original weights to
#     produce one normal, self-contained model again.
#
#   - "Quantization" (8-bit / 4-bit loading): normally, each individual
#     number (weight) in a neural network is stored using 16 or 32 bits of
#     precision. Quantization stores each number using fewer bits (e.g. 8 or
#     4) instead, which makes the model take up much less memory and run
#     faster, at the cost of a small amount of numerical precision/accuracy.
#     This is very useful for running huge models on GPUs with limited
#     memory.
#
#   - "PEFT" (Parameter-Efficient Fine-Tuning): the general name for the
#     family of techniques that LoRA belongs to; also the name of the
#     popular Hugging Face Python library (`peft`) that implements them,
#     used directly in this file (`from peft import PeftModel`).
# =============================================================================


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
# These four all come from Hugging Face's `transformers` library, the most
# widely used Python library for working with pretrained language models:
#   - AutoTokenizer: automatically figures out and loads the RIGHT
#     tokenizer for a given model, without you needing to know its exact
#     class name in advance.
#   - AutoModelForCausalLM: same idea, but automatically loads the right
#     "causal language model" class (a model that predicts the next word
#     given previous words) for a given model path.
#   - AutoConfig: automatically loads a model's saved configuration/settings
#     (layer sizes, vocabulary size, special multimodal settings, etc.)
#     without needing to know its exact class name.
#   - BitsAndBytesConfig: settings object used to configure 8-bit/4-bit
#     quantized loading (see "Quantization" explanation above), powered by
#     the `bitsandbytes` library.
import torch
from llava.model import *
# Imports every class defined in the llava.model package (including
# LlavaLlamaForCausalLM, LlavaMptForCausalLM, LlavaMistralForCausalLM, and
# the LlavaMetaModel/LlavaMetaForCausalLM classes from the previous file you
# sent) using a wildcard import, so they can all be referenced by name
# directly in this file without a package prefix.
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# Special placeholder text strings (like "<im_patch>") explained in detail
# in the previous file's comments -- used here to make sure the tokenizer
# being loaded actually has these special tokens registered in its
# vocabulary.


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    # PLAIN-LANGUAGE EXPLANATION OF THE ARGUMENTS:
    #   model_path: where the (possibly fine-tuned/LoRA) model lives -- a
    #     local folder path or a Hugging Face Hub repo name.
    #   model_base: if this model is a LoRA fine-tune or "delta" on top of
    #     another model, this is the path/name of that ORIGINAL base model
    #     needed to reconstruct the full thing. Left as None if model_path
    #     already points at a complete, self-contained model.
    #   model_name: a string used mainly to figure out (by checking whether
    #     it contains substrings like "llava", "lora", "mpt", "mistral")
    #     WHICH loading code path to take. Notably this is usually just the
    #     folder name, used as a naming convention/hint rather than being
    #     read from inside the checkpoint itself.
    #   load_8bit / load_4bit: whether to load the model using reduced
    #     numeric precision to save GPU memory (see "Quantization" above).
    #   device_map: tells Hugging Face how to spread the model's layers
    #     across available devices (GPUs/CPU). "auto" lets the library
    #     figure out the best arrangement automatically, which is
    #     especially useful for very large models that might not fit
    #     entirely on one GPU.
    #   device: which single device to use if you're not using the "auto"
    #     device_map behavior (e.g. "cuda" for a GPU, "cpu" for no GPU).
    #   use_flash_attn: whether to use "Flash Attention", a faster,
    #     more memory-efficient way of computing the attention mechanism
    #     inside transformer models (an optimized low-level implementation
    #     of the same underlying math, not a different model architecture).
    #   **kwargs: catches any additional keyword arguments the caller
    #     passes in (like your custom `my_model_config`, referenced later in
    #     this function) so they can be forwarded straight into the
    #     model-loading calls below without this function needing to know
    #     about every possible extra setting in advance.
    kwargs = {"device_map": device_map, **kwargs}
    # Build a dictionary of settings that will later be handed straight to
    # Hugging Face's `.from_pretrained(...)` loading functions. Starting it
    # off with device_map and then spreading in **kwargs means any custom
    # arguments the caller passed (like my_model_config) get included too.

    print('kwargs: ', kwargs)

    if device != "cuda":
        # If a specific non-GPU device was requested (e.g. "cpu"), override
        # the "auto" device_map with an explicit instruction: put the
        # ENTIRE model (represented by the empty-string key "") onto that
        # one device, rather than letting Hugging Face auto-distribute
        # layers across multiple devices.
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            # Even when the STORED weights are compressed to 4 bits, the
            # actual math computed during a forward pass is still done in a
            # more precise format (float16 = 16-bit "half precision"
            # floating point numbers) to keep results reasonably accurate;
            # this setting controls that computation precision.
            bnb_4bit_use_double_quant=True,
            # "Double quantization": an extra trick that also compresses
            # the small helper numbers (quantization constants) used to
            # decompress the 4-bit weights, squeezing out a little more
            # memory savings on top of the main 4-bit compression.
            bnb_4bit_quant_type='nf4'
            # "nf4" = "4-bit NormalFloat", a specific numerical format
            # designed to represent typical neural-network weight values
            # (which tend to cluster in a bell-curve/normal distribution
            # around zero) more accurately than a plain, evenly-spaced 4-bit
            # format would.
        )
    else:
        # If no quantization is requested, just load weights in a
        # standard, moderately compact format: float16 ("half precision",
        # 16 bits per number) rather than the even larger default of
        # float32 (32 bits per number) -- a very common middle-ground
        # choice for running (not training) large models efficiently.
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        # --------------------------------------------------------------
        # The model's NAME (folder name / hub repo name) contains "llava",
        # so we go down the multimodal-model loading path rather than
        # treating it as a plain text-only language model.
        # --------------------------------------------------------------
        if 'lora' in model_name.lower() and model_base is None:
            # The name suggests this is a LoRA fine-tune (which, remember,
            # only contains a small set of ADD-ON weights, not a full
            # model), but no base model was given to combine it with --
            # warn the user, since loading will likely fail or produce a
            # broken/incomplete model without it.
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            # ============================================================
            # CASE 1: Loading a LoRA fine-tuned LLaVA model, which requires
            # combining a base model with the small saved set of LoRA
            # changes AND this project's own custom "non-LoRA trainable"
            # weights (the mm_projector / mm_scene_projector networks,
            # which -- unlike the frozen base model -- were fully trained
            # from scratch alongside the LoRA weights, so they're saved
            # separately as ordinary, complete weights rather than as
            # LoRA's small add-on matrices).
            # ============================================================
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            # Load just the CONFIGURATION (settings/shapes/hyperparameters)
            # of the fine-tuned model -- not its weights yet -- from the
            # LoRA checkpoint folder. We need this config so the base model
            # gets constructed with the exact same shapes/settings that the
            # LoRA fine-tune expects.

            #print('lora_cfg_pretrained: ', lora_cfg_pretrained)

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            # Load the tokenizer from the BASE model (not the LoRA
            # checkpoint), since the base model is what actually defines
            # the starting vocabulary. use_fast=False picks the pure-Python
            # tokenizer implementation instead of the faster Rust-based
            # one -- sometimes needed for compatibility with certain custom
            # tokenizer behavior.
            print('Loading LLaVA from base model...')
            print('kwargs["my_model_config"]: ', kwargs["my_model_config"])
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            # Load the actual BASE model's weights (e.g. plain Vicuna/LLaMA
            # weights, or a previous full LLaVA checkpoint), but tell it to
            # use the fine-tuned model's config (lora_cfg_pretrained) --
            # this ensures things like vocabulary size and multimodal
            # settings match what the LoRA fine-tune expects, even though
            # the raw NUMBERS being loaded right now are still the base
            # model's original numbers. low_cpu_mem_usage=True is a
            # Hugging Face optimization that avoids creating a full extra
            # temporary copy of the model in regular CPU memory while
            # loading, which matters a lot for huge multi-billion-parameter
            # models.


            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            # "lm_head" ("language model head") is the final layer of the
            # model that converts its last internal vector into a score for
            # every possible next word in the vocabulary. Its "out_features"
            # is the vocabulary size (how many possible words/tokens it can
            # predict), and "in_features" is the size of the internal
            # vector it reads (e.g. 4096).
            if model.lm_head.weight.shape[0] != token_num:
                # If the fine-tuned config's vocabulary size doesn't match
                # what the base model currently has (this happens because
                # fine-tuning added new special tokens like "<im_patch>",
                # growing the vocabulary), resize both the final
                # prediction layer AND the initial word-embedding lookup
                # table to have the right number of rows, filled with
                # empty/uninitialized placeholder values for now -- they'll
                # get overwritten with real trained numbers a few lines
                # down when the saved LoRA/non-LoRA weights are loaded in.
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            # MY_DEBUG
            print('model_path: ', model_path)
            # /home/hsukuangc/my_co_llm_driver/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v6sdebugmm_scene_and_object/checkpoint-100
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                # "non_lora_trainables.bin" is exactly what its name says:
                # a saved file containing all the weights that were trained
                # DURING this fine-tuning run but that AREN'T part of
                # LoRA's small add-on matrices -- in this project, that
                # mainly means the mm_projector and mm_scene_projector
                # networks (which are trained fully/normally, not via
                # LoRA), plus the resized word-embedding/output rows for
                # any newly added special tokens.
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                # MY_DEBUG
                # inference code hit here
                #assert False
            else:
                # this is probably from HF Hub
                # If the file isn't found as a plain local file, assume
                # model_path is actually the name of a Hugging Face Hub
                # repository instead of a local folder, and download the
                # file from there.
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
                # MY_DEBUG
                assert False
                # This crash is a safety-net: it seems this project's
                # models are always loaded from a local folder rather than
                # the Hugging Face Hub, so if this branch is ever actually
                # reached, it's a sign something unexpected happened (e.g. a
                # wrong or missing local path) and it's better to fail
                # loudly than silently download from an unintended source.

            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            # Strip a "base_model." prefix from the saved weight names if
            # present -- this prefix gets added automatically by the PEFT
            # library when it wraps a model for LoRA training, and needs to
            # be removed so the plain, unwrapped model's parameter names
            # match up correctly when loading these weights back in.
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            # Similarly strip a doubled-up "model.model." prefix down to
            # just "model." if present -- this kind of naming mismatch
            # commonly happens because of how PyTorch nests modules inside
            # each other and how the training wrapper renamed things.

            # MY_CODE
            # check how this non_lora_trainables load works
            # ------------------------------------------------------------
            # Debug/sanity-check block: before actually loading the new
            # weights in, print out the projector layers' current
            # (base-model, not-yet-updated) shapes and the SUM of all their
            # numbers -- a quick way to visually confirm, a few lines down,
            # that the sum changes after loading (proving the new weights
            # really did get applied, rather than the loading silently
            # doing nothing).
            # ------------------------------------------------------------
            for name, param in model.named_parameters():
                if 'mm_projector' in name or 'mm_scene_projector' in name:
                    print('name: ', name)
                    print('param.data.shape: ', param.data.shape)
                    # name:  model.mm_projector.0.weight
                    # param.data.shape:  torch.Size([4096, 1024])
                    # name:  model.mm_scene_projector.0.weight
                    # param.data.shape:  torch.Size([4096, 3072])
                    # These shapes confirm what earlier files described:
                    # the regular projector maps 1024-number vectors to
                    # 4096-number vectors, while the scene projector maps
                    # 3072-number vectors (the "deep" scene features) to
                    # 4096-number vectors. PyTorch stores a linear layer's
                    # weight matrix as [output_size, input_size], which is
                    # why the numbers appear in this [4096, X] order.

                    print("torch.sum(param.data): ", torch.sum(param.data))

            model.load_state_dict(non_lora_trainables, strict=False)
            # Actually copy the loaded projector/embedding weights into the
            # live model. strict=False means "don't error out just because
            # this dictionary doesn't contain weights for EVERY single
            # parameter in the model" -- which is expected here, since
            # non_lora_trainables.bin only contains a small subset (the
            # projectors and a few embedding rows), not the whole model.

            print('After loading non_lora_trainables\n')

            for name, param in model.named_parameters():
                if 'mm_projector' in name or 'mm_scene_projector' in name:
                    print('name: ', name)
                    #print('param.data.shape: ', param.data.shape)
                    print("torch.sum(param.data): ", torch.sum(param.data))
                    # Printed again after loading, so you can compare
                    # before/after sums by eye in the console output and
                    # confirm the projector weights actually changed (i.e.
                    # really were loaded from the checkpoint, not left as
                    # random/base-model values).

            #assert False

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            # PeftModel.from_pretrained wraps the base model with the
            # actual small LoRA add-on matrices loaded from model_path.
            # After this line, the model behaves as the fine-tuned version,
            # but internally it's still structured as "base model + small
            # separate LoRA adjustment layers", not one single merged set
            # of weights yet.
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            # merge_and_unload() mathematically combines the small LoRA
            # add-on matrices directly INTO the base model's original
            # weight matrices (this works because of how LoRA's math is
            # designed -- the add-on effect can be added straight onto the
            # original weights), and then removes/discards the now-redundant
            # separate LoRA wrapper. The result is one normal, plain,
            # self-contained model again, just like any other -- convenient
            # for running inference without needing the PEFT library
            # involved anymore.
            print('Model is loaded...')
        elif model_base is not None:
            # ============================================================
            # CASE 2: A base model was given, but the folder name doesn't
            # contain "lora" -- this is meant for the scenario where only
            # the mm_projector was trained/saved separately (an even
            # smaller, simpler kind of "delta" than a full LoRA fine-tune),
            # while everything else stays as the original base model.
            # Note: this whole branch is guarded by `assert False` further
            # down (see "not tested"), meaning it is NOT currently used or
            # verified to work in this project -- it's inherited from the
            # original LLaVA codebase for a scenario this project doesn't
            # actually exercise.
            # ============================================================
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                # "MPT" is a different open-source language model family
                # (from MosaicML) that LLaVA can also be built on top of,
                # as an alternative to LLaMA/Vicuna. It needs a small
                # extra step: copying over a Python file that defines its
                # custom configuration class, since MPT models rely on
                # "trust_remote_code" (custom Python code bundled with the
                # checkpoint) rather than being fully standardized inside
                # the transformers library itself.
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            # MY_CODE
            # not tested
            assert False
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            # ============================================================
            # CASE 3: No base model given at all -- model_path already
            # points at a COMPLETE, self-contained LLaVA model (e.g. one
            # that was already merged, or trained fully from scratch
            # rather than via LoRA). Just load it directly, picking the
            # right specific model class based on which underlying
            # language-model family its name mentions (MPT, Mistral, or
            # the default LLaMA-based LlavaLlamaForCausalLM).
            # ============================================================
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
                # This is almost certainly the branch this project actually
                # uses for straightforward inference: model_path pointing
                # directly at a complete LLaVA checkpoint (whether that's
                # one that was merged from LoRA earlier, or a full-parameter
                # fine-tune), with model_base left as None.
    else:
        # ================================================================
        # model_name doesn't contain "llava" at all -- fall back to
        # treating this as a PLAIN, text-only language model with no
        # vision/multimodal capability whatsoever (no vision tower, no
        # projector). Not really relevant to this project, but kept here
        # as a general-purpose fallback inherited from the original LLaVA
        # codebase (e.g. useful for comparing a multimodal model's answers
        # against a plain text-only baseline model).
        # ================================================================
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None
    # An "image processor" (sometimes called an image transform/feature
    # extractor) is a helper object that resizes, crops, and numerically
    # normalizes raw images into the exact tensor format the vision tower
    # expects -- the image equivalent of what a tokenizer does for text.
    # It defaults to None for plain non-LLaVA text models, since they don't
    # process images at all.

    if 'llava' in model_name.lower():
        # ----------------------------------------------------------------
        # Final multimodal setup steps, run regardless of which of the
        # three CASE branches above was taken -- these make sure the
        # tokenizer's special tokens and the vision tower are fully ready
        # to use.
        # ----------------------------------------------------------------
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        # Same idea as initialize_vision_tokenizer in the previous file:
        # make sure the tokenizer has the special multimodal tokens
        # registered, and make sure the model's embedding table has enough
        # rows to match. This is done again here (separately from
        # training-time setup) because at INFERENCE time, we're loading a
        # tokenizer fresh and need to make sure it ends up in the exact
        # same state (same added tokens, same vocabulary size) as it was
        # during training, so token IDs line up correctly with what the
        # model learned.

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            # Recall from the previous file that the vision tower can be
            # constructed with delay_load=True, meaning its architecture
            # exists but its actual pretrained weights haven't been loaded
            # into memory yet. This checks for that and loads the real
            # weights now, since we're about to actually use the model.
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            # If we're not letting Hugging Face auto-distribute the model
            # across devices, explicitly move the vision tower onto the
            # chosen device (device_map here, somewhat confusingly, is
            # being reused as a plain device identifier in this branch)
            # and switch it to float16 for efficient, consistent
            # precision.
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor
        # Grab the vision tower's paired image processor so the caller of
        # this function can use it to correctly preprocess any raw images
        # before feeding them to the model (not really used by this
        # project's own sensor-data experiments, but still set up here
        # since it's part of the general LLaVA loading routine).

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
        # "Context length" is the maximum total number of tokens (words +
        # inserted image/sensor tokens combined) the model can handle in a
        # single input sequence at once. If the loaded model's config
        # doesn't specify this explicitly, fall back to a conservative
        # default of 2048 tokens.

    return tokenizer, model, image_processor, context_len
    # Hand back everything the caller needs to actually use the model:
    # the tokenizer (for converting text to/from token IDs), the fully
    # loaded model itself, the image processor (for preprocessing images,
    # if relevant), and the maximum context length it supports.
