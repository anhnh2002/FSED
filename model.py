import torch
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
from configs import parse_arguments
from llm2vec import LLM2Vec
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, List

args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")  # type: ignore


class LLM2VecED(nn.Module):
    def __init__(self, class_num=args.class_num + 1, input_map=False):
        super().__init__()
        self.backbone = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            device_map=device,
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=args.max_seqlen,
        )

        self.backbone.model = self.initialize_peft(
            self.backbone.model
        )

        self.input_dim = self.backbone.model.config.hidden_size
        self.fc = nn.Linear(self.input_dim, class_num, dtype=torch.bfloat16)

    
    def load_checkpoint(self, cp_path: str):
        self.backbone = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            device_map=device,
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=args.max_seqlen,
        )

        self.backbone.model = PeftModel.from_pretrained(self.backbone.model, cp_path)

        self.fc.load_state_dict(torch.load(cp_path + "/state_dict.pth"), strict=False)

    def save_checkpoint(self, cp_path: str):
        self.backbone.model.save_pretrained(cp_path)
        torch.save(self.fc.state_dict(), cp_path + "/state_dict.pth")
    
    def forward(self, x, masks, span=None, aug=None):
        # x = self.backbone(x) #TODO: test use
        return_dict = {}
        backbone_output = self.backbone.model(input_ids = x, attention_mask = masks)
        x, pooled_feat = backbone_output[0], backbone_output[1]
        context_feature = x.view(-1, x.shape[-1])
        return_dict['reps'] = weighted_average_pooling(x, masks).clone()
        if span != None:
            outputs, trig_feature = [], []
            for i in range(len(span)):
                
                opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
                # x = x_cdt.permute(1, 0, 2) 
                trig_feature.append(opt)
            trig_feature = torch.cat(trig_feature)
        outputs = self.fc(trig_feature)
        return_dict['outputs'] = outputs
        return_dict['context_feat'] = context_feature
        return_dict['trig_feat'] = trig_feature
        # if args.single_label:
        #     return_outputs = self.fc(enc_out_feature).view(-1, args.class_num + 1)
        # else:
        #     return_outputs = self.fc(feature)
        if aug is not None:
            feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
            outputs_aug = self.fc(feature_aug)
            return_dict['feature_aug'] = feature_aug
            return_dict['outputs_aug'] = outputs_aug
        return return_dict

    def forward_backbone(self, x, masks):
        x = self.backbone.model(input_ids=x, attention_mask = masks)
        x = x.last_hidden_state
        return x


    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

class BertED(nn.Module):
    def __init__(self, class_num=args.class_num + 1, input_map=False):
        super().__init__()
        self.backbone = BertModel.from_pretrained(args.backbone)
        if not args.no_freeze_bert:
            print("Freeze bert parameters")
            for _, param in list(self.backbone.named_parameters()):
                param.requires_grad = False
        else:
            print("Update bert parameters")
        self.is_input_mapping = input_map
        self.input_dim = self.backbone.config.hidden_size
        self.fc = nn.Linear(self.input_dim, class_num)
        if self.is_input_mapping:
            self.map_hidden_dim = 512 # 512 is implemented by the paper
            self.map_input_dim =  self.input_dim * 2
            self.input_map = nn.Sequential(
                nn.Linear(self.map_input_dim, self.map_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.map_hidden_dim, self.map_hidden_dim),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.map_hidden_dim, class_num)

    def forward(self, x, masks, span=None, aug=None):
        # x = self.backbone(x) #TODO: test use
        return_dict = {}
        backbone_output = self.backbone(x, attention_mask = masks)
        x, pooled_feat = backbone_output[0], backbone_output[1]
        context_feature = x.view(-1, x.shape[-1])
        return_dict['reps'] = x[:, 0, :].clone()
        if span != None:
            outputs, trig_feature = [], []
            for i in range(len(span)):
                if self.is_input_mapping:
                    x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
                    x_cdt = x_cdt.permute(1, 0, 2)
                    x_cdt = x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
                    opt = self.input_map(x_cdt)
                else:
                    opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
                    # x = x_cdt.permute(1, 0, 2) 
                trig_feature.append(opt)
            trig_feature = torch.cat(trig_feature)
        outputs = self.fc(trig_feature)
        return_dict['outputs'] = outputs
        return_dict['context_feat'] = context_feature
        return_dict['trig_feat'] = trig_feature
        # if args.single_label:
        #     return_outputs = self.fc(enc_out_feature).view(-1, args.class_num + 1)
        # else:
        #     return_outputs = self.fc(feature)
        if aug is not None:
            feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
            outputs_aug = self.fc(feature_aug)
            return_dict['feature_aug'] = feature_aug
            return_dict['outputs_aug'] = outputs_aug
        return return_dict

    def forward_backbone(self, x, masks):
        x = self.backbone(x, attention_mask = masks)
        x = x.last_hidden_state
        return x

    def forward_input_map(self, x):
        return self.input_map(x)



def weighted_average_pooling(x, attention_mask):
    # Ensure inputs are the correct shape
    assert x.dim() == 3, "x should be 3-dimensional (B, N, H)"
    assert attention_mask.dim() == 2, "attention_mask should be 2-dimensional (B, N)"
    assert x.shape[0] == attention_mask.shape[0], "Batch sizes should match"
    assert x.shape[1] == attention_mask.shape[1], "Sequence lengths should match"

    # Convert attention_mask to float and add a dimension to match x
    mask = attention_mask.float().unsqueeze(-1)  # Shape: (B, N, 1)

    # Normalize the mask
    mask_sum = mask.sum(dim=1, keepdim=True)  # Shape: (B, 1, 1)
    mask_normalized = mask / (mask_sum + 1e-9)  # Add small epsilon to avoid division by zero

    # Apply the weighted average pooling
    result = torch.sum(x * mask_normalized, dim=1)  # Shape: (B, H)

    return result