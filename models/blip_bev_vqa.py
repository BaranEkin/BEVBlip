import torch
from torch import nn

import transformers

transformers.logging.set_verbosity_error()

from models.blip import init_tokenizer
from models.vit import VisionTransformer
from models.med import BertConfig, BertModel, BertLMHeadModel


class BLIP_BEV_VQA(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        bev_size=50,
        bev_dim=256,
        visual_width=768,
        use_vit=True,
        use_det=False,
        use_obj=False
    ):
        super().__init__()
        self.device = "cuda"
        self.tokenizer = init_tokenizer()
        self.use_vit = use_vit
        self.use_det = use_det
        self.use_obj = use_obj

        self.visual_width = visual_width
        
        # BEV --------------------------------------------------------------------------------------
        self.bev_size = bev_size
        self.bev_dim = bev_dim
        
        # Visual Encoder (ViT) ---------------------------------------------------------------------
        if self.use_vit:
            
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

        # BEV features from Detection Locations ----------------------------------------------------
        if self.use_det:
            self.det_proj = nn.Linear(bev_dim, visual_width)

        # Object Features from BEVFormer Decoder  
        if self.use_obj:
            self.obj_proj = nn.Linear(bev_dim, visual_width)

        # Text Encoder -----------------------------------------------------------------------------
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = self.visual_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_width = self.text_encoder.config.hidden_size

        # Text Decoder ----------------------------------------------------------------------------
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = self.visual_width
        self.text_decoder = BertLMHeadModel(config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        # BLIP Checkpoint ---------------------------------------------------------------------------
        checkpoint = torch.load("/workspace/thesis/ckpts/model_base_capfilt_large.pth")
        self.load_state_dict(checkpoint["model"], strict=False)
        print("BLIP checkpoint loaded.")

    def forward(self, question, answer, bev=None, det=None, obj=None):
        visual_embeds = None

        # Get visual embeds -----------------------------------------------------------------------
        if self.use_vit:
            assert bev is not None, "No bev feature"
            bev_embeds = self.vis_encoder(
                bev.view(-1, self.bev_size, self.bev_size, self.bev_dim).permute(0, 3, 1, 2)
            )
            if visual_embeds is not None:
                visual_embeds = torch.cat([visual_embeds, bev_embeds], dim=1)
            else:
                visual_embeds = bev_embeds

        if self.use_det:
            assert det is not None, "No det feature"
            det_embeds = self.det_proj(det)
            
            if visual_embeds is not None:
                visual_embeds = torch.cat([visual_embeds, det_embeds], dim=1)
            else:
                visual_embeds = det_embeds

        if self.use_obj:
            assert obj is not None, "No obj feature"
            obj_embeds = self.obj_proj(obj)

            if visual_embeds is not None:
                visual_embeds = torch.cat([visual_embeds, obj_embeds], dim=1)
            else:
                visual_embeds = obj_embeds

        visual_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(self.device)
        bs = visual_embeds.size(0)

        question = self.tokenizer(
            question,
            padding="longest",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        ).to(self.device)
        question.input_ids[:, 0] = self.tokenizer.enc_token_id

        answer = self.tokenizer(
            answer,
            padding="longest",
            truncation=True,
            max_length=300,
            return_tensors="pt",
        ).to(self.device)
        answer.input_ids[:, 0] = self.tokenizer.bos_token_id
        answer_targets = answer.input_ids.masked_fill(
            answer.input_ids == self.tokenizer.pad_token_id, -100
        )

        # Get question embeds ---------------------------------------------------------
        question_output = self.text_encoder(
            question.input_ids,
            attention_mask=question.attention_mask,
            encoder_hidden_states=visual_embeds,
            encoder_attention_mask=visual_atts,
            return_dict=True,
        )

        question_states = []
        question_atts = []
        n = [1] * bs
        for b, n in enumerate(n):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [question.attention_mask[b]] * n
        question_states = torch.stack(question_states, 0)
        question_atts = torch.stack(question_atts, 0)
        
        # Get answer output -----------------------------------------------------------
        answer_output = self.text_decoder(
            answer.input_ids,
            attention_mask=answer.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=answer_targets,
            return_dict=True,
            reduction="none",
        )

        loss = answer_output.loss.mean()
        return loss

    def generate(
        self,
        question,
        max_length=300,
        min_length=1,
        top_p=0.9,
        temperature=0.8,
        bev=None,
        det=None,
        obj=None,
    ):
        visual_embeds = None
        
        # Get visual embeds -----------------------------------------------------------------------
        if self.use_vit:
            assert bev is not None, "No bev feature"
            bev_embeds = self.vis_encoder(
                bev.view(-1, self.bev_size, self.bev_size, self.bev_dim).permute(0, 3, 1, 2)
            )
            if visual_embeds is not None:
                visual_embeds = torch.cat([visual_embeds, bev_embeds], dim=1)
            else:
                visual_embeds = bev_embeds

        if self.use_det:
            assert det is not None, "No det feature"
            det_embeds = self.det_proj(det)
            
            if visual_embeds is not None:
                visual_embeds = torch.cat([visual_embeds, det_embeds], dim=1)
            else:
                visual_embeds = det_embeds

        if self.use_obj:
            assert obj is not None, "No obj feature"
            obj_embeds = self.obj_proj(obj)

            if visual_embeds is not None:
                visual_embeds = torch.cat([visual_embeds, obj_embeds], dim=1)
            else:
                visual_embeds = obj_embeds

        visual_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(self.device)
        bs = visual_embeds.size(0)

        # Get question embeds ---------------------------------------------------------
        question = self.tokenizer(
            question,
            padding="longest",
            truncation=True,
            max_length=300,
            return_tensors="pt",
        ).to(self.device)
        question.input_ids[:, 0] = self.tokenizer.enc_token_id

        question_output = self.text_encoder(
            question.input_ids,
            attention_mask=question.attention_mask,
            encoder_hidden_states=visual_embeds,
            encoder_attention_mask=visual_atts,
            return_dict=True,
        )

        question_states = question_output.last_hidden_state
        question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(
            question_states.device
        )
        model_kwargs = {
            "encoder_hidden_states": question_states,
            "encoder_attention_mask": question_atts,
        }

        bos_ids = torch.full(
            (bs, 1), fill_value=self.tokenizer.bos_token_id, device=self.device
        )

        # nucleus sampling
        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.1,
            **model_kwargs
        )

        answers = []
        for output in outputs:
            answer = self.tokenizer.decode(output, skip_special_tokens=True)
            answers.append(answer)
        return answers
        
