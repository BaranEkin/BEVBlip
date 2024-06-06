import torch
import torch.nn.functional as F
import transformers
from torch import nn
from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import init_tokenizer
from models.vit import VisionTransformer

transformers.logging.set_verbosity_error()


class BLIP_BEV_Pretrain(nn.Module):
    def __init__(
        self,
        med_config="configs/bert_config.json",
        blip_ckpt="ckpts/model_base_capfilt_large.pth",
        bev_size=50,
        bev_dim=256,
        embed_dim=256,
        visual_width=768,
        queue_size=10,
        momentum=0.995,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            blip_ckpt (str): path for the blip_pretrain model checkpoint to load
            bev_size (int): input bev's spatial size (height or width)
            bev_dim (int): input bev's feature size
            embed_dim (int): common embedding dim size for visual and text features to be projected onto
            queue_size (int): queue size
            momentum (float): momentum parameter for momentum encoders
        """

        super().__init__()
        self.device = "cuda"

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.tokenizer = init_tokenizer()

        # BEV --------------------------------------------------------------------------------------
        self.bev_size = bev_size
        self.bev_dim = bev_dim

        # Visual Encoder ---------------------------------------------------------------------------
        self.visual_width = visual_width
        self.vit_patch_size = 5
        self.vit_depth = 3
        self.vit_num_heads = 12

        self.vis_encoder = VisionTransformer(
            img_size=self.bev_size,
            patch_size=self.vit_patch_size,
            in_chans=self.bev_dim,
            embed_dim=self.visual_width,
            depth=self.vit_depth,
            num_heads=self.vit_num_heads,
            use_grad_checkpointing=False,
            ckpt_layer=0,
            drop_path_rate=0,
        )

        # Text Encoder -----------------------------------------------------------------------------
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = self.visual_width
        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased", config=encoder_config, add_pooling_layer=False
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_width = self.text_encoder.config.hidden_size

        # Text Decoder ----------------------------------------------------------------------------
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = self.visual_width
        self.text_decoder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=decoder_config
        )
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        tie_encoder_decoder_weights(
            self.text_encoder, self.text_decoder.bert, "", "/attention"
        )

        # ITM Head ---------------------------------------------------------------------------------
        self.itm_head = nn.Linear(self.text_width, 2)

        # Projectors -------------------------------------------------------------------------------
        self.embed_dim = embed_dim

        self.vis_proj = nn.Linear(self.visual_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        # Momentum models --------------------------------------------------------------------------
        self.momentum = momentum

        self.vis_proj_m = nn.Linear(self.visual_width, self.embed_dim)
        self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)

        self.vis_encoder_m = VisionTransformer(
            img_size=self.bev_size,
            in_chans=self.bev_dim,
            patch_size=self.vit_patch_size,
            embed_dim=self.visual_width,
            depth=self.vit_depth,
            num_heads=self.vit_num_heads,
            use_grad_checkpointing=False,
            ckpt_layer=0,
            drop_path_rate=0,
        )

        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)

        self.model_pairs = [
            [self.vis_encoder, self.vis_encoder_m],
            [self.vis_proj, self.vis_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        # Queues -----------------------------------------------------------------------------------
        self.queue_size = queue_size

        self.register_buffer("bev_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.bev_queue = nn.functional.normalize(self.bev_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        # BLIP Checkpoint ---------------------------------------------------------------------------
        self.load_blip_ckpt(blip_ckpt)

        # Prompt ------------------------------------------------------------------------------------
        self.prompt = ""
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, bev, caption, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        bev_embeds = self.vis_encoder(
            bev.view(-1, self.bev_size, self.bev_size, self.bev_dim).permute(0, 3, 1, 2)
        )

        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(self.device)
        bev_feat = F.normalize(self.vis_proj(bev_embeds[:, 0, :]), dim=-1)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=70,
            return_tensors="pt",
        ).to(self.device)
        text_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            bev_embeds_m = self.vis_encoder_m(
                bev.view(-1, self.bev_size, self.bev_size, self.bev_dim).permute(
                    0, 3, 1, 2
                )
            )
            bev_feat_m = F.normalize(self.vis_proj_m(bev_embeds_m[:, 0, :]), dim=-1)
            bev_feat_all = torch.cat(
                [bev_feat_m.t(), self.bev_queue.clone().detach()], dim=1
            )

            text_output_m = self.text_encoder_m(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_feat_m = F.normalize(
                self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1
            )
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = bev_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ bev_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(self.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_i2t = bev_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ bev_feat_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(bev_feat_m, text_feat_m)

        # BEV-Text Matching ----------------------------------------------------------------
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positive BEV-text pair
        bs = bev_embeds.size(0)
        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=bev_embeds,
            encoder_attention_mask=bev_atts,
            return_dict=True,
        )
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)

        # select a negative BEV for each text
        bev_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            bev_embeds_neg.append(bev_embeds[neg_idx])
        bev_embeds_neg = torch.stack(bev_embeds_neg, dim=0)

        # select a negative text for each BEV
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        bev_embeds_all = torch.cat([bev_embeds_neg, bev_embeds], dim=0)
        bev_atts_all = torch.cat([bev_atts, bev_atts], dim=0)

        output_neg = self.text_encoder(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=bev_embeds_all,
            encoder_attention_mask=bev_atts_all,
            return_dict=True,
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        # LM ---------------------------------------------------------------------------------------
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_output = self.text_decoder(
            decoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=bev_embeds,
            encoder_attention_mask=bev_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        loss_lm = decoder_output.loss
        return loss_ita, loss_itm, loss_lm

    def generate(
        self,
        bev,
        max_length=100,
        min_length=5,
        top_p=0.9,
    ):
        bs = bev.size(0)
        bev_embeds = self.vis_encoder(
            bev.view(-1, self.bev_size, self.bev_size, self.bev_dim).permute(0, 3, 1, 2)
        )

        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(self.device)
        model_kwargs = {
            "encoder_hidden_states": bev_embeds,
            "encoder_attention_mask": bev_atts,
        }

        prompt = [""] * bs  # batch size
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        # nucleus sampling
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.1,
            **model_kwargs,
        )

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, bev_feat, text_feat):
        # gather keys before updating queue
        bev_feats = concat_all_gather(bev_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = bev_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.bev_queue[:, ptr:ptr + batch_size] = bev_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def load_blip_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        ckpt_dict = ckpt["model"]

        self_dict = self.state_dict()
        ckpt_dict = {k: v for k, v in ckpt_dict.items() if k in self_dict}
        del ckpt_dict["text_queue"]
        self_dict.update(ckpt_dict)
        self.load_state_dict(self_dict)
        print(f"BLIP checkpoint loaded from: {ckpt_path}")


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


from typing import List


def tie_encoder_decoder_weights(
    encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str
):
    uninitialized_encoder_weights: List[str] = []
    assert decoder.__class__ == encoder.__class__

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + " is tied")
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set(
                [module_name + "/" + sub_name for sub_name in encoder_modules.keys()]
            )
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(
                        decoder_modules[decoder_name],
                        type(encoder_modules[encoder_name]),
                    ) and len(encoder_modules) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a "
                        "circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(
        decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key
    )
