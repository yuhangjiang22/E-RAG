import torch
from torch import nn
import numpy as np
from typing import List
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers import PretrainedConfig
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

BertLayerNorm = torch.nn.LayerNorm

class ERAGConfig(PretrainedConfig):

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        hidden_dropout_prob=0.1,
        num_labels=6,
        tokenizer_len=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.hidden_dropout_prob=hidden_dropout_prob
        self.num_labels = num_labels
        self.tokenizer_len = tokenizer_len

class ERAG(PreTrainedModel):

    config_class = ERAGConfig

    def __init__(self, config):
        super(ERAG, self).__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        )
        self.hf_config = hf_config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(hf_config.hidden_size)
        self.post_init()
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(hf_config.hidden_size, config.num_labels)
        self.doc_classifier = nn.Linear(hf_config.hidden_size, config.num_labels)
        self.input_linear = nn.Linear(hf_config.hidden_size * 2, hf_config.hidden_size)
        self.doc_linear = nn.Linear(hf_config.hidden_size * 3, hf_config.hidden_size)
        self.tokenizer_len = config.tokenizer_len
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.input_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
            add_pooling_layer=False
        )
        self.documents_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
            add_pooling_layer=False
        )
        if self.tokenizer_len:
            self.input_encoder.resize_token_embeddings(self.tokenizer_len)
            self.documents_encoder.resize_token_embeddings(self.tokenizer_len)


    # def _init_weights(self, module):
    #     """Initialize the weights"""
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self):
        self.input_encoder.gradient_checkpointing_enable()
        self.documents_encoder.gradient_checkpointing_enable()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            labels = None,
            sub_idx = None,
            obj_idx = None,
            doc_input_ids: torch.LongTensor = None,
            doc_input_mask: torch.Tensor = None,
            doc_type_ids: torch.Tensor = None,
            return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.hf_config.use_return_dict

        description_outputs = self.documents_encoder(
            doc_input_ids,
            attention_mask=doc_input_mask,
            token_type_ids=doc_type_ids,
            return_dict=return_dict,
        )

        # batch_size*num_types x seq_length x hidden_size
        doc_sequence_output = description_outputs[0]

        outputs = self.input_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]
        batch_size, seq_length, _ = sequence_output.size()
        # batch_size*num_types x seq_length x hidden_size
        batch_size, doc_seq_length, _ = doc_sequence_output.size()

        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)

        cls_rep = torch.cat([a[0].unsqueeze(0) for a in doc_sequence_output])
        cls_rep = torch.cat((rep, cls_rep), dim=1)

        rep = self.input_linear(rep)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)

        cls_rep = self.doc_linear(cls_rep)
        cls_rep = self.layer_norm(cls_rep)
        cls_rep = self.dropout(cls_rep)

        logits = self.classifier(rep)
        doc_logits = self.doc_classifier(cls_rep)

        logits_probs = F.softmax(logits, dim=-1)

        # Compute entropy of logits to measure how "decisive" they are
        logits_entropy = -torch.sum(logits_probs * torch.log(logits_probs + 1e-12), dim=-1)

        # Normalize entropy to be between 0 and 1 (optional, for smoothness)
        max_entropy = torch.log(torch.tensor(self.num_labels, dtype=torch.float32))
        normalized_entropy = self.beta * logits_entropy / max_entropy

        # Define dynamic_alpha based on logits entropy (lower entropy = more decisive)
        dynamic_alpha = 1 - normalized_entropy  # if logits_entropy is low, dynamic_alpha will be high

        # Combine logits and doc_logits using dynamic_alpha
        combined_logits = dynamic_alpha.unsqueeze(1) * logits + (1 - dynamic_alpha.unsqueeze(1)) * doc_logits
        # combined_logits = doc_logits

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(combined_logits.view(-1, self.num_labels), labels.view(-1))
            loss2 = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.alpha * loss1 + (1 - self.alpha) * loss2
            return loss
        else:
            return combined_logits


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_size = hidden_size

        # Cross-attention between document and input representations
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(0.1)

    def forward(self, query_tensor, key_value_tensor, attention_mask=None):
        # Calculate attention scores
        query_layer = self.query(query_tensor)
        key_layer = self.key(key_value_tensor)
        value_layer = self.value(key_value_tensor)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.hidden_size ** 0.5)

        # Apply attention mask (if any)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)

        # Get the final cross-attended representation
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer


class ERAGWithCrossAttention(PreTrainedModel):

    config_class = ERAGConfig

    def __init__(self, config):
        super(ERAGWithCrossAttention, self).__init__(config)

        hf_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)

        self.input_encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path, add_pooling_layer=False)
        self.documents_encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path, add_pooling_layer=False)

        self.cross_attention_layer = CrossAttentionLayer(hf_config.hidden_size)
        self.layer_norm = nn.LayerNorm(hf_config.hidden_size * 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hf_config.hidden_size * 2, config.num_labels)

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                labels=None,
                sub_idx=None,
                obj_idx=None,
                doc_input_ids: torch.LongTensor = None,
                doc_input_mask: torch.Tensor = None,
                doc_type_ids: torch.Tensor = None,
                return_dict: bool = None,):
        # Encode input text
        input_outputs = self.input_encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=return_dict
        )
        sequence_output = input_outputs.last_hidden_state

        # Encode documents
        doc_outputs = self.documents_encoder(
            doc_input_ids, attention_mask=doc_input_mask, token_type_ids=doc_type_ids, return_dict=return_dict
        )
        doc_sequence_output = doc_outputs.last_hidden_state

        # Cross-attention between input and document encoder outputs
        cross_attended_output = self.cross_attention_layer(sequence_output, doc_sequence_output)

        # Combine representations (e.g., by summing or concatenation)
        combined_rep = torch.cat([sequence_output, cross_attended_output], dim=-1)
        combined_rep = self.layer_norm(combined_rep)
        combined_rep = self.dropout(combined_rep)

        # Get subject-object representations for classification
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(combined_rep, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(combined_rep, obj_idx)])

        # Final relation representation (sub-object interaction)
        relation_rep = torch.cat((sub_output, obj_output), dim=1)

        # Classify relation
        logits = self.classifier(relation_rep)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return loss
        else:
            return logits