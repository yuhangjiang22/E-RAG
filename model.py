import torch
from torch import nn
import numpy as np
from typing import List
import logging
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers import PretrainedConfig
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.beta_value = nn.Parameter(torch.tensor(0.5))

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
        normalized_entropy = self.beta_value * logits_entropy / max_entropy

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
            return loss2
        else:
            return logits


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
        self.classifier = nn.Linear(hf_config.hidden_size * 4, config.num_labels)
        self.pure_classifier = nn.Linear(hf_config.hidden_size * 2, config.num_labels)
        self.beta_value = nn.Parameter(torch.tensor(0.5))
        self.num_labels = config.num_labels

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

        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        pure_logits = self.pure_classifier(rep)

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

        logits_probs = F.softmax(pure_logits, dim=-1)

        # Compute entropy of logits to measure how "decisive" they are
        logits_entropy = -torch.sum(logits_probs * torch.log(logits_probs + 1e-12), dim=-1)

        # Normalize entropy to be between 0 and 1 (optional, for smoothness)
        max_entropy = torch.log(torch.tensor(self.num_labels, dtype=torch.float32))
        normalized_entropy = self.beta_value * logits_entropy / max_entropy

        # Define dynamic_alpha based on logits entropy (lower entropy = more decisive)
        dynamic_alpha = 1 - normalized_entropy  # if logits_entropy is low, dynamic_alpha will be high

        # Combine logits and doc_logits using dynamic_alpha
        combined_logits = dynamic_alpha.unsqueeze(1) * pure_logits + (1 - dynamic_alpha.unsqueeze(1)) * logits

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            loss = loss_fct(combined_logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return combined_logits


class DocumentAttention(nn.Module):
    def __init__(self, hidden_size):
        super(DocumentAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(0.1)

    def forward(self, query_tensor, key_value_tensor, doc_mask=None):
        # Compute attention scores (dot-product)
        query_layer = self.query(query_tensor).unsqueeze(1)  # batch_size x 1 x hidden_size
        key_layer = self.key(key_value_tensor)  # batch_size x doc_seq_len x hidden_size
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)).squeeze(1)  # batch_size x doc_seq_len

        # Apply attention mask (if provided)
        if doc_mask is not None:
            attention_scores = attention_scores.masked_fill(doc_mask == 0, -1e9)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch_size x doc_seq_len
        attention_probs = self.attention_dropout(attention_probs)

        # Weighted sum of document representations
        value_layer = self.value(key_value_tensor)
        attended_document_rep = torch.matmul(attention_probs.unsqueeze(1), value_layer).squeeze(
            1)  # batch_size x hidden_size

        return attended_document_rep, attention_probs


class ERAGWithDocumentAttention(PreTrainedModel):

    config_class = ERAGConfig

    def __init__(self, config):
        super(ERAGWithDocumentAttention, self).__init__(config)

        hf_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)

        self.input_encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path, add_pooling_layer=False)
        self.documents_encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path, add_pooling_layer=False)

        self.document_attention = DocumentAttention(hf_config.hidden_size)
        self.layer_norm = nn.LayerNorm(hf_config.hidden_size * 2)
        self.combined_rep_layer_norm = nn.LayerNorm(hf_config.hidden_size * 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classifiers
        self.input_classifier = nn.Linear(hf_config.hidden_size * 2, config.num_labels)  # For input-only logits
        self.doc_classifier = nn.Linear(hf_config.hidden_size * 3, config.num_labels)  # For document-only logits
        self.combined_classifier = nn.Linear(hf_config.hidden_size * 2, config.num_labels)  # For combined logits

        # Dynamic Relevance Network: a small feedforward network that predicts a dynamic relevance score
        self.relevance_net = nn.Sequential(
            nn.Linear(hf_config.hidden_size * 2, hf_config.hidden_size),
            nn.ReLU(),
            nn.Linear(hf_config.hidden_size, 1),
            nn.Sigmoid()  # Output relevance score between 0 and 1
        )

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

        # Use attention to weight the document's relevance dynamically
        input_cls_rep = sequence_output[:, 0, :]  # batch_size x hidden_size

        # Compute the attention-weighted document representation
        attended_doc_rep, attention_probs = self.document_attention(input_cls_rep, doc_sequence_output,
                                                                    doc_mask=doc_input_mask)

        # Get subject-object representations for input
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])

        # Final relation representation (subject-object interaction)
        input_relation_rep = torch.cat((sub_output, obj_output), dim=-1)

        # Combine input and document representations
        combined_rep = torch.cat((input_relation_rep, attended_doc_rep), dim=-1)

        # Apply LayerNorm, dropout, and classifier
        combined_rep = self.combined_rep_layer_norm(combined_rep)
        combined_rep = self.dropout(combined_rep)

        input_relation_rep = self.layer_norm(input_relation_rep)
        input_relation_rep = self.dropout(input_relation_rep)

        # Compute input-only and document-only logits
        input_logits = self.input_classifier(input_relation_rep)  # Input-based logits
        doc_logits = self.doc_classifier(combined_rep)  # Document-based logits

        # Dynamic relevance score prediction (use input and document representations)
        relevance_input = torch.cat([input_cls_rep, attended_doc_rep], dim=-1)  # Combine input and doc [CLS] reps
        dynamic_relevance_score = self.relevance_net(relevance_input).squeeze(
            -1)  # Predict relevance score per instance

        # Compute the combined logits based on the dynamic relevance score
        combined_logits = dynamic_relevance_score.unsqueeze(1) * doc_logits + (
                    1 - dynamic_relevance_score.unsqueeze(1)) * input_logits

        if labels is not None:
            # Loss calculation
            loss_fct = nn.CrossEntropyLoss()
            combined_loss = loss_fct(combined_logits.view(-1, self.combined_classifier.out_features), labels.view(-1))
            input_loss = loss_fct(input_logits.view(-1, self.input_classifier.out_features), labels.view(-1))
            doc_loss = loss_fct(doc_logits.view(-1, self.doc_classifier.out_features), labels.view(-1))

            # Total loss (using the dynamic relevance score to weigh input and document losses)
            # loss = dynamic_relevance_score * doc_loss + (1 - dynamic_relevance_score) * input_loss + combined_loss
            return combined_loss
        else:
            return combined_logits

class MultiHeadDocumentAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadDocumentAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"

        # Linear layers for query, key, and value
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output linear layer to combine all heads
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(0.1)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim).
        Transpose the result so the shape is (batch_size, num_heads, seq_len, head_dim)
        """
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query_tensor, key_value_tensor, doc_mask=None):
        batch_size = query_tensor.size(0)
        doc_seq_len = key_value_tensor.size(1)

        # Linear projections
        query_layer = self.split_heads(self.query(query_tensor), batch_size)  # (batch_size, num_heads, 1, head_dim)
        key_layer = self.split_heads(self.key(key_value_tensor), batch_size)  # (batch_size, num_heads, doc_seq_len, head_dim)
        value_layer = self.split_heads(self.value(key_value_tensor), batch_size)  # (batch_size, num_heads, doc_seq_len, head_dim)

        # Compute scaled dot-product attention for each head
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.head_dim ** 0.5)
        # (batch_size, num_heads, 1, doc_seq_len)

        # Apply mask if provided
        if doc_mask is not None:
            doc_mask = doc_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, doc_seq_len)
            attention_scores = attention_scores.masked_fill(doc_mask == 0, -1e6)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (batch_size, num_heads, 1, doc_seq_len)
        attention_probs = self.attention_dropout(attention_probs)

        # Weighted sum of values
        attended_heads = torch.matmul(attention_probs, value_layer)  # (batch_size, num_heads, 1, head_dim)

        # Reshape back to (batch_size, seq_len, hidden_size) by concatenating the heads
        attended_heads = attended_heads.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final linear layer to combine all heads
        attended_document_rep = self.out(attended_heads).squeeze(1)  # (batch_size, hidden_size)

        return attended_document_rep, attention_probs

class ERAGWithDocumentMHAttention(PreTrainedModel):

    config_class = ERAGConfig

    def __init__(self, config):
        super(ERAGWithDocumentMHAttention, self).__init__(config)

        hf_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)
        self.hf_config = hf_config

        self.input_encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path, add_pooling_layer=False)
        self.documents_encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path, add_pooling_layer=False)

        self.document_attention = MultiHeadDocumentAttention(hf_config.hidden_size, 12)
        # self.document_attention = DocumentAttention(hf_config.hidden_size)
        self.layer_norm = nn.LayerNorm(hf_config.hidden_size * 2)
        self.combined_rep_layer_norm = nn.LayerNorm(hf_config.hidden_size * 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classifiers
        self.input_classifier = nn.Linear(hf_config.hidden_size * 2, config.num_labels)  # For input-only logits
        self.doc_classifier = nn.Linear(hf_config.hidden_size * 3, config.num_labels)  # For document-only logits
        self.combined_classifier = nn.Linear(hf_config.hidden_size * 2, config.num_labels)  # For combined logits

        # Dynamic Relevance Network: a small feedforward network that predicts a dynamic relevance score
        self.relevance_net = nn.Sequential(
            nn.Linear(hf_config.hidden_size * 2, hf_config.hidden_size),
            nn.ReLU(),
            nn.Linear(hf_config.hidden_size, 1),
            nn.Sigmoid()  # Output relevance score between 0 and 1
        )

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
        batch_size, seq_len, hidden_size = sequence_output.shape

        # Encode documents
        doc_outputs = self.documents_encoder(
            doc_input_ids, attention_mask=doc_input_mask, token_type_ids=doc_type_ids, return_dict=return_dict
        )
        doc_sequence_output = doc_outputs.last_hidden_state

        # Use attention to weight the document's relevance dynamically
        input_cls_rep = sequence_output[:, 0, :]  # batch_size x hidden_size

        num_docs = int(doc_sequence_output.size(0) / batch_size)
        logger.info('num_docs: {}'.format(num_docs))
        # batch_size x num_docs x seq_len x hidden_size
        doc_sequence_output = doc_sequence_output.view(batch_size, num_docs, seq_len, hidden_size)
        logger.info('doc_sequence_output: {}'.format(doc_sequence_output))

        doc_input_mask = doc_input_mask.view(batch_size, num_docs, seq_len)
        logger.info('doc_input_mask: {}'.format(doc_input_mask))
        # batch_size x num_docs * seq_len x hidden_size
        doc_sequence_output = doc_sequence_output.view(batch_size, seq_len * num_docs, hidden_size)
        logger.info('doc_sequence_output: {}'.format(doc_sequence_output))

        doc_input_mask = doc_input_mask.view(batch_size, seq_len * num_docs)
        logger.info('doc_input_mask: {}'.format(doc_input_mask))
        # Compute the attention-weighted document representation
        attended_doc_rep, attention_probs = self.document_attention(input_cls_rep, doc_sequence_output,
                                                                    doc_mask=doc_input_mask)
        logger.info('attended_doc_rep: {}'.format(attended_doc_rep))

        # Get subject-object representations for input
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])

        # Final relation representation (subject-object interaction)
        input_relation_rep = torch.cat((sub_output, obj_output), dim=-1)

        # Combine input and document representations
        combined_rep = torch.cat((input_relation_rep, attended_doc_rep), dim=-1)

        # Apply LayerNorm, dropout, and classifier
        combined_rep = self.combined_rep_layer_norm(combined_rep)
        combined_rep = self.dropout(combined_rep)

        input_relation_rep = self.layer_norm(input_relation_rep)
        input_relation_rep = self.dropout(input_relation_rep)

        # Compute input-only and document-only logits
        input_logits = self.input_classifier(input_relation_rep)  # Input-based logits
        doc_logits = self.doc_classifier(combined_rep)  # Document-based logits

        # Dynamic relevance score prediction (use input and document representations)
        relevance_input = torch.cat([input_cls_rep, attended_doc_rep], dim=-1)  # Combine input and doc [CLS] reps
        dynamic_relevance_score = self.relevance_net(relevance_input).squeeze(
            -1)  # Predict relevance score per instance

        # Compute the combined logits based on the dynamic relevance score
        combined_logits = dynamic_relevance_score.unsqueeze(1) * doc_logits + (
                    1 - dynamic_relevance_score.unsqueeze(1)) * input_logits

        if labels is not None:
            # Loss calculation
            loss_fct = nn.CrossEntropyLoss()
            combined_loss = loss_fct(combined_logits.view(-1, self.combined_classifier.out_features), labels.view(-1))
            # input_loss = loss_fct(input_logits.view(-1, self.input_classifier.out_features), labels.view(-1))
            # doc_loss = loss_fct(doc_logits.view(-1, self.doc_classifier.out_features), labels.view(-1))

            # Total loss (using the dynamic relevance score to weigh input and document losses)
            # loss = dynamic_relevance_score * doc_loss + (1 - dynamic_relevance_score) * input_loss + combined_loss
            return combined_loss, combined_logits
        else:
            return combined_logits
