import argparse
import logging
import os
import random
import time
import json
import sys
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from utils import generate_relation_data, decode_sample_id
from const import task_rel_labels, task_ner_labels
from model import ERAG, ERAGWithCrossAttention, ERAGWithDocumentAttention, ERAGWithDocumentMHAttention, ERAGWithSelfRAG, ERAGWithSelfRAG2, ERAGConfig

biored_type_map = {
    'DiseaseOrPhenotypicFeature': 'disease',
    'SequenceVariant': 'variant',
    'GeneOrGeneProduct': 'gene',
    'ChemicalEntity': 'drug'
}

class InputFeatures(object):

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 sub_idx,
                 obj_idx,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

def add_description_words(tokenizer, documents):

    def add_words(d):
        if type(d) == list:
            for l in d:
                add_words(l.spilt())
        else:
            for w in d:
                if w not in tokenizer.vocab:
                    unk_words.append(w)

    unk_words = []
    add_words(documents)
    tokenizer.add_tokens(unk_words)

CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>' % label)
        new_tokens.append('<SUBJ_END=%s>' % label)
        new_tokens.append('<OBJ_START=%s>' % label)
        new_tokens.append('<OBJ_END=%s>' % label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>' % label)
        new_tokens.append('<OBJ=%s>' % label)
    new_tokens = [token.lower() for token in new_tokens]
    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d' % len(tokenizer))

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_f1(preds, labels):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1,
                'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': n_gold}

def convert_examples_to_features(args, examples, label2id, tokenizer, special_tokens, documents, unused_tokens=True):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    """

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        return special_tokens[w]

    def get_documents(entity, documents, args, type=None):
        num_docs = args.num_docs
        if 'entity' in documents[0]:
            key_type = 'entity'
        else:
            key_type = 'entity_pair'
        found = False
        for el in documents:
            curr_key = el[key_type]
            if curr_key == entity:
                if args.task == 'biored':
                    if el['type'] == biored_type_map[type]:
                        found = True
                        break
                else:
                    found = True
                    break
        if not found:
            logger.info('No relevant documents found for {} with type {}'.format(entity, biored_type_map[type]))
            return []

        documents = el['texts'][:num_docs]
        documents = [i['content'] for i in documents]
        return documents

    def get_biored_documents(entity, doc_id, documents, args, type=None):
        num_docs = args.num_docs
        entity_info = read_dict('biored/entityid2string.json')
        doc_entity_info = read_dict('biored/doc2entity.json')
        entity = update_entity(entity, entity_type=biored_type_map[type], doc_id=doc_id, entity_info=entity_info, doc_entity_info=doc_entity_info)
        if 'entity' in documents[0]:
            key_type = 'entity'
        else:
            key_type = 'entity_pair'
        found = False
        for el in documents:
            curr_key = el[key_type]
            if curr_key == entity and el['type'] == biored_type_map[type]:
                found = True
                break
            if curr_key.replace(' ', '') == entity.replace(' ', '') and el['type'] == biored_type_map[type]:
                found = True
                break
        if not found:
            logger.info('No relevant documents found for {} with type {}'.format(entity, biored_type_map[type]))
            return []

        documents = el['texts'][:num_docs]
        documents = [i['content'] for i in documents]
        return documents

    def get_doc_tokens(doc):
        doc_tokens = [CLS]
        doc = doc.split()
        for token in doc:
            for sub_token in tokenizer.tokenize(token):
                doc_tokens.append(sub_token)
        doc_tokens.append(SEP)
        # doc_max_tokens = max(doc_max_tokens, len(doc_tokens))
        if len(doc_tokens) > max_seq_length:
            doc_tokens = doc_tokens[:max_seq_length]

        doc_segment_ids = [0] * len(doc_tokens)
        doc_input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        doc_input_mask = [1] * len(doc_input_ids)
        padding = [0] * (max_seq_length - len(doc_input_ids))
        doc_input_ids += padding
        doc_input_mask += padding
        doc_segment_ids += padding
        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert doc_input_ids[0] == 2

        return doc_input_ids, doc_input_mask, doc_segment_ids

    max_seq_length = args.max_seq_length
    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    doc_max_tokens = 0
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]

        SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'])
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'])

        for i, token in enumerate(example['token']):
            if i == example['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START_NER)
            if i == example['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START_NER)
            for sub_token in tokenizer.tokenize(token):
                tokens.append(sub_token)
            if i == example['subj_end']:
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                tokens.append(OBJECT_END_NER)
        tokens.append(SEP)

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            if sub_idx >= max_seq_length:
                sub_idx = 0
            if obj_idx >= max_seq_length:
                obj_idx = 0
        else:
            num_fit_examples += 1


        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example['relation']]
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        subj = ' '.join(example['token'][example['subj_start']:example['subj_end'] + 1])
        obj = ' '.join(example['token'][example['obj_start']:example['obj_end'] + 1])
        if 'pair' in args.document_path:
            docs = get_documents(f'{subj}|{obj}', documents, args)
        else:
            if args.task == 'biored':
                doc_id = example['docid']
                docs = get_biored_documents(subj, doc_id, documents, args, example['subj_type']) + get_biored_documents(obj, doc_id, documents, args, example['obj_type'])
            else:
                docs = get_documents(subj, documents, args, example['subj_type']) + get_documents(obj, documents, args, example['obj_type'])

        docs_input_ids, docs_input_mask, docs_segment_ids = [], [], []

        for doc in docs:
            doc_input_ids, doc_input_mask, doc_segment_ids = get_doc_tokens(doc)
            docs_input_ids.append(doc_input_ids)
            docs_input_mask.append(doc_input_mask)
            docs_segment_ids.append(doc_segment_ids)

        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example['id']))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("doc_input_ids: %s" % " ".join([str(x) for x in docs_input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("docs_input_mask: %s" % " ".join([str(x) for x in docs_input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("docs_segment_ids: %s" % " ".join([str(x) for x in docs_segment_ids]))
                logger.info("label: %s (id = %d)" % (example['relation'], label_id))
                logger.info("sub_idx, obj_idx: %d, %d" % (sub_idx, obj_idx))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              sub_idx=sub_idx,
                              obj_idx=obj_idx,
                              doc_input_ids=docs_input_ids,
                              doc_input_mask=docs_input_mask,
                              doc_segment_ids=docs_segment_ids))
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("Max #tokens: %d"%max_tokens)
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features

def evaluate(model, device, eval_dataloader, eval_label_ids):
    model.eval()
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx, doc_input_ids, doc_input_mask, doc_type_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)

        doc_input_ids = doc_input_ids.to(device)
        doc_input_mask = doc_input_mask.to(device)
        doc_type_ids = doc_type_ids.to(device)

        batch_size, num_docs, doc_seq_length = doc_input_ids.size()
        doc_input_ids = doc_input_ids.reshape(batch_size * num_docs, doc_seq_length)
        doc_input_mask = doc_input_mask.reshape(batch_size * num_docs, doc_seq_length)
        doc_type_ids = doc_type_ids.reshape(batch_size * num_docs, doc_seq_length)

        with torch.no_grad():
            logits = model(input_ids,
                           input_mask,
                           segment_ids,
                           labels=None,
                           sub_idx=sub_idx,
                           obj_idx=obj_idx,
                           doc_input_ids=doc_input_ids,
                           doc_input_mask=doc_input_mask,
                           doc_type_ids=doc_type_ids,
                           return_dict=True)
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds[0], axis=1)
    result = compute_f1(preds, eval_label_ids.numpy())
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())

    return preds, result


def print_pred_json(eval_data, eval_examples, preds, id2label, output_file):
    rels = dict()
    for ex, pred in zip(eval_examples, preds):
        doc_sent, sub, obj = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        if pred != 0:
            rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], id2label[pred]])

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d' % (doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))

    logger.info('Output predictions to %s..' % (output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))


def setseed(seed):
    random.seed(seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_trained_model(output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info('Saving model to %s' % output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_vocabulary(output_dir)

def read_docs(path):
    files = []
    with open(path, 'r') as file:
        for line in file:
            json_line = json.loads(line.strip())
            files.append(json_line)
    return files

def normalize(el, el_type, doc_id, doc_entity_info, entity_info):

    def map_entity_text_to_id(doc_id, ner_type, text):
        entity_ids = doc_entity_info[doc_id]
        for id_ in entity_ids:
            endwith = id_[id_.find('@@')+2:]
            if endwith == ner_type:
                potential_texts = entity_info[id_]
                if text in potential_texts:
                    # c += 1
                    return max(potential_texts, key=len)
                else:
                    if text.replace(' ', '') in [t.replace(' ', '') for t in potential_texts]:
                        # c += 1
                        return max(potential_texts, key=len)
        return None

    new_ent = map_entity_text_to_id(doc_id, el_type, el)

    return new_ent

def update_entity(entity, entity_type, doc_id, doc_entity_info, entity_info):

    entity = normalize(entity, entity_type, doc_id, doc_entity_info, entity_info)

    return entity




def read_dict(path):
    with open(path, 'r') as file:
        files = json.load(file)
    return files

def main(args):

    setseed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # train set
    if args.do_train:
        train_dataset, train_examples, train_nrel = generate_relation_data(args.train_file, context_window=args.context_window, task=args.task)
    # dev set
    if (args.do_eval and args.do_train) or (args.do_eval and not (args.eval_test)):
        eval_dataset, eval_examples, eval_nrel = generate_relation_data(
            os.path.join(args.file_dir, args.dev_file),
            context_window=args.context_window, task=args.task)
    # test set
    if args.eval_test:
        test_dataset, test_examples, test_nrel = generate_relation_data(
            os.path.join(args.file_dir, args.test_file),
            context_window=args.context_window, task=args.task)

    documents_path = args.document_path
    relevant_documents = read_docs(documents_path)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(
        device, n_gpu))

    # get label_list
    if os.path.exists(os.path.join(args.output_dir, 'label_list.json')):
        with open(os.path.join(args.output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    config = ERAGConfig(pretrained_model_name_or_path=args.model,
                        hidden_dropout_prob=0.1,
                        num_labels=num_labels,
                        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    # add_description_words(tokenizer, relevant_documents)

    if args.add_new_tokens:
        add_marker_tokens(tokenizer, task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    if args.do_eval and (args.do_train or not (args.eval_test)):
        eval_features = convert_examples_to_features(
            args, eval_examples, label2id, tokenizer, special_tokens, relevant_documents, unused_tokens=not(args.add_new_tokens))

        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)

        all_doc_input_ids = torch.tensor([f.doc_input_ids for f in eval_features], dtype=torch.long)
        all_doc_input_mask = torch.tensor([f.doc_input_mask for f in eval_features], dtype=torch.long)
        all_doc_segment_ids = torch.tensor([f.doc_segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids,
                                  all_input_mask,
                                  all_segment_ids,
                                  all_label_ids,
                                  all_sub_idx,
                                  all_obj_idx,
                                  all_doc_input_ids,
                                  all_doc_input_mask,
                                  all_doc_segment_ids,)

        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids

    if args.do_train:
        train_features = convert_examples_to_features(
            args, train_examples, label2id, tokenizer, special_tokens, relevant_documents, unused_tokens=not(args.add_new_tokens))

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in train_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in train_features], dtype=torch.long)

        all_doc_input_ids = torch.tensor([f.doc_input_ids for f in train_features], dtype=torch.long)
        all_doc_input_mask = torch.tensor([f.doc_input_mask for f in train_features], dtype=torch.long)
        all_doc_segment_ids = torch.tensor([f.doc_segment_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids,
                                  all_input_mask,
                                  all_segment_ids,
                                  all_label_ids,
                                  all_sub_idx,
                                  all_obj_idx,
                                  all_doc_input_ids,
                                  all_doc_input_mask,
                                  all_doc_segment_ids,)

        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)

        lr = args.learning_rate

        if 'PubMedBERT' not in config.pretrained_model_name_or_path:
            config.tokenizer_len = len(tokenizer)

        if args.cross_att:
            model = ERAGWithCrossAttention(config)
        elif args.doc_mhatt:
            model = ERAGWithDocumentMHAttention(config)
        elif args.doc_att:
            model = ERAGWithDocumentAttention(config)
        elif args.self_rag:
            model = ERAGWithSelfRAG(config)
        elif args.self_rag2:
            model = ERAGWithSelfRAG2(config)
        else:
            model = ERAG(config)

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=not(args.bertadam))
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    int(num_train_optimization_steps * args.warmup_proportion),
                                                    num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0

        for epoch in range(int(args.num_train_epochs)):
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in enumerate(train_batches):
                # batch_size, _ = batch[0].size()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx, doc_input_ids, doc_input_mask, doc_segment_ids = batch
                batch_size, num_docs, doc_seq_length = doc_input_ids.size()

                doc_input_ids = doc_input_ids.reshape(batch_size * num_docs, doc_seq_length)
                doc_input_mask = doc_input_mask.reshape(batch_size * num_docs, doc_seq_length)
                doc_segment_ids = doc_segment_ids.reshape(batch_size * num_docs, doc_seq_length)

                loss = model(input_ids, input_mask, segment_ids, label_ids,
                             sub_idx, obj_idx, doc_input_ids, doc_input_mask, doc_segment_ids, return_dict=True)
                # logger.info('loss: {}'.format(loss))
                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # logger.info('tr_loss: {}'.format(tr_loss))

                if (step + 1) % eval_step == 0:
                    logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                        epoch, step + 1, len(train_batches),
                               time.time() - start_time, tr_loss / nb_tr_steps))
                    save_model = False
                    if args.do_eval:
                        preds, result = evaluate(model=model,
                                                 device=device,
                                                 eval_dataloader=eval_dataloader,
                                                 eval_label_ids=eval_label_ids,
                                                 )
                        model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = args.train_batch_size

                        if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                            best_result = result
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                            save_trained_model(args.output_dir, model, tokenizer)

    if args.do_eval:
        logger.info(special_tokens)
        if args.eval_test:
            eval_dataset = test_dataset
            eval_examples = test_examples
            eval_features = convert_examples_to_features(
                args, test_examples, label2id, tokenizer, special_tokens, relevant_documents, unused_tokens=not(args.add_new_tokens))
            logger.info(special_tokens)
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
            all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)

            all_doc_input_ids = torch.tensor([f.doc_input_ids for f in eval_features], dtype=torch.long)
            all_doc_input_mask = torch.tensor([f.doc_input_mask for f in eval_features], dtype=torch.long)
            all_doc_segment_ids = torch.tensor([f.doc_segment_ids for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids,
                                       all_input_mask,
                                       all_segment_ids,
                                       all_label_ids,
                                       all_sub_idx,
                                       all_obj_idx,
                                       all_doc_input_ids,
                                       all_doc_input_mask,
                                       all_doc_segment_ids)

            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
            eval_label_ids = all_label_ids
        if args.cross_att:
            model = ERAGWithCrossAttention.from_pretrained(args.output_dir, num_labels=num_labels)
        elif args.doc_mhatt:
            model = ERAGWithDocumentMHAttention.from_pretrained(args.output_dir, num_labels=num_labels)
        elif args.doc_att:
            model = ERAGWithDocumentAttention.from_pretrained(args.output_dir, num_labels=num_labels)
        elif args.self_rag:
            model = ERAGWithSelfRAG.from_pretrained(args.output_dir, num_labels=num_labels)
        elif args.self_rag2:
            model = ERAGWithSelfRAG2.from_pretrained(args.output_dir, num_labels=num_labels)
        else:
            model = ERAG.from_pretrained(args.output_dir, num_labels=num_labels)
        model.to(device)
        preds, result = evaluate(model=model,
                                 device=device,
                                 eval_dataloader=eval_dataloader,
                                 eval_label_ids=eval_label_ids,
                                 )

        logger.info('*** Evaluation Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        print_pred_json(eval_dataset, eval_examples, preds, id2label,
                        os.path.join(args.output_dir, args.prediction_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_file", default=None, type=str, help="The path of the training data.")
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")
    parser.add_argument("--file_dir", type=str, default=None,
                        help="The directory of the prediction files of the entity model")
    parser.add_argument("--dev_file", type=str, default="dev.json",
                        help="The entity prediction file of the dev set")
    parser.add_argument("--test_file", type=str, default="test.json",
                        help="The entity prediction file of the test set")
    parser.add_argument("--prediction_file", type=str, default="predictions.json",
                        help="The prediction filename for the relation model")
    parser.add_argument('--task', type=str, default=None, required=True,
                        choices=['scierc', 'chemprot', 'chemprot_5', 'biored', 'cdr'])
    parser.add_argument('--context_window', type=int, default=0)
    parser.add_argument('--add_new_tokens', action='store_true',
                        help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
    parser.add_argument('--train_num_examples', type=int, default=None,
                        help="Number of training instances.")
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help="hidden drop out rate.")
    parser.add_argument('--num_docs', type=int, default=1)
    parser.add_argument("--document_path", default=None, type=str, help="The path to the relevant documents.")
    parser.add_argument('--cross_att', action='store_true')
    parser.add_argument('--doc_att', action='store_true')
    parser.add_argument('--doc_mhatt', action='store_true')
    parser.add_argument('--self_rag', action='store_true')
    parser.add_argument('--self_rag2', action='store_true')
    # parser.add_argument('--doc_type', type=str, default=None, required=True,
                        # choices=['doc', 'doc_pair'])
    args = parser.parse_args()
    main(args)













