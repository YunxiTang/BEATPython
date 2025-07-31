import flax.linen as nn


class LanguageTokenizer(nn.Module):
    """
    Language tokenizer that embeds text input IDs into continuous language embeddings. Supports pre-trained HF models.

     Args:
         num_tokens (int): Number of output tokens (not enforced).
         encoder (str, optional): Optional HuggingFace AutoModel name for encoding input IDs.
         finetune_encoder (bool, optional): Optional finetune last layers of the language model.
    """

    encoder: str = None
    finetune_encoder: bool = False
    proper_pad_mask: bool = True

    def setup(self):
        if self.encoder is not None:
            from transformers import AutoConfig, FlaxAutoModel, FlaxT5EncoderModel

            config = AutoConfig.from_pretrained(self.encoder)
            if "t5" in self.encoder:
                self.hf_model = FlaxT5EncoderModel(config).module
            else:
                self.hf_model = FlaxAutoModel.from_config(config).module

    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        if "language_instruction" not in tasks:
            logging.warning("No language inputs found. Skipping tokenizer entirely.")
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        if not isinstance(tasks["language_instruction"], (jax.Array, np.ndarray)):
            assert self.encoder is not None, (
                "Received language tokens but no encoder specified."
            )
            tokens = self.hf_model(**tasks["language_instruction"]).last_hidden_state
        else:
            # add a # tokens dimension to language
            if tasks["language_instruction"].ndim == 2:
                tokens = tasks["language_instruction"][:, None, :]
            else:
                tokens = tasks["language_instruction"]

        if not self.finetune_encoder:
            tokens = jax.lax.stop_gradient(tokens)

        # TODO: incorporate padding info from language tokens here too
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens,
                tasks.get("pad_mask_dict", None),
                ("language_instruction",),
            )
        else:
            pad_mask = jnp.ones(tokens.shape[:-1])

        return TokenGroup(tokens, pad_mask)
