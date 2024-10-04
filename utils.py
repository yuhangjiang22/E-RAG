import logging

from data_structures import Dataset

logger = logging.getLogger('root')


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens,
                                 tokenized_id2description, unused_tokens=False):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    """
    CLS = "[CLS]"
    SEP = "[SEP]"

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        return special_tokens[w]

    def get_description_input(description_tokens):
        description_tokens = [CLS] + description_tokens
        description_tokens = [subject if word == '@subject@' else word for word in description_tokens]
        description_tokens = [object if word == '@object@' else word for word in description_tokens]
        description_tokens = [item for sublist in description_tokens for item in
                              (sublist if isinstance(sublist, list) else [sublist])]
        description_tokens.append(SEP)

        des_sub_idx = description_tokens.index(SUBJECT_START_NER)
        des_obj_idx = description_tokens.index(OBJECT_START_NER)
        descriptions_sub_idx.append(des_sub_idx)
        descriptions_obj_idx.append(des_obj_idx)

        description_input_ids = tokenizer.convert_tokens_to_ids(description_tokens)
        description_type_ids = [0] * len(description_tokens)
        description_input_mask = [1] * len(description_input_ids)
        padding = [0] * (max_seq_length - len(description_input_ids))
        description_input_ids += padding
        description_input_mask += padding
        description_type_ids += padding

        assert len(description_input_ids) == max_seq_length
        assert len(description_input_mask) == max_seq_length
        assert len(description_type_ids) == max_seq_length

        return description_input_ids, description_input_mask, description_type_ids

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]

        SUBJECT_START_NER = get_special_token("SUBJ_START=%s" % example['subj_type'])
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s" % example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s" % example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s" % example['obj_type'])

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
                sub_idx_end = len(tokens)
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                obj_idx_end = len(tokens)
                tokens.append(OBJECT_END_NER)
        tokens.append(SEP)

        subject = tokens[sub_idx:sub_idx_end + 1]
        object = tokens[obj_idx:obj_idx_end + 1]

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

        descriptions_input_ids = []
        descriptions_input_mask = []
        descriptions_type_ids = []
        descriptions_sub_idx = []
        descriptions_obj_idx = []

        for _, description_tokens_list in tokenized_id2description.items():

            description_tokens = description_tokens_list[0]
            description_input_ids, description_input_mask, description_type_ids = get_description_input(description_tokens)
            descriptions_input_ids.append(description_input_ids)
            descriptions_input_mask.append(description_input_mask)
            descriptions_type_ids.append(description_type_ids)


        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example['id']))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example['relation'], label_id))
                logger.info("sub_idx, obj_idx: %d, %d" % (sub_idx, obj_idx))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          sub_idx=sub_idx,
                          obj_idx=obj_idx,
                          descriptions_input_ids=descriptions_input_ids,
                          descriptions_input_mask=descriptions_input_mask,
                          descriptions_type_ids=descriptions_type_ids,
                          descriptions_sub_idx=descriptions_sub_idx,
                          descriptions_obj_idx=descriptions_obj_idx))
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("Max #tokens: %d" % max_tokens)
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                                                                       num_fit_examples * 100.0 / len(examples),
                                                                       max_seq_length))
    return features


def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id.split('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))

    return doc_sent, sub, obj


def generate_relation_data(entity_data, context_window=0, task=None):
    """
    Prepare data for the relation model
    If training: set use_gold = True
    """
    biored_entity_pairs = [['DiseaseOrPhenotypicFeature', 'SequenceVariant'],
                           ['DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'],
                           ['DiseaseOrPhenotypicFeature', 'ChemicalEntity'],
                           ['ChemicalEntity', 'ChemicalEntity'],
                           ['ChemicalEntity', 'SequenceVariant'],
                           ['SequenceVariant', 'SequenceVariant'],
                           ['GeneOrGeneProduct', 'ChemicalEntity'],
                           ['GeneOrGeneProduct', 'GeneOrGeneProduct']]
    logger.info('Generate relation data from %s' % (entity_data))
    data = Dataset(entity_data)

    nner, nrel = 0, 0
    max_sentsample = 0
    samples = []
    num_null = 0
    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []

            nner += len(sent.ner)
            nrel += len(sent.relations)

            sent_ner = sent.ner

            gold_ner = {}
            for ner in sent.ner:
                gold_ner[ner.span] = ner.label

            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label

            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text

            if context_window > 0:
                add_left = (context_window - len(sent.text)) // 2
                add_right = (context_window - len(sent.text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            captured = []
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    type_pair = [sub.label, obj.label]
                    if task == 'chemprot' or task == 'chemprot_5':
                        if sub.label == 'CHEMICAL' and obj.label == 'GENE':  # Subject can only be CHEMICAL and object can only be GENE
                            label = gold_rel.get((sub.span, obj.span), 'no_relation')
                            sample = {}
                            sample['docid'] = doc._doc_key
                            sample['id'] = '%s@%d::(%d,%d)-(%d,%d)' % (
                            doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc,
                            obj.span.end_doc)
                            sample['relation'] = label
                            sample['subj_start'] = sub.span.start_sent + sent_start
                            sample['subj_end'] = sub.span.end_sent + sent_start
                            sample['subj_type'] = sub.label
                            sample['obj_start'] = obj.span.start_sent + sent_start
                            sample['obj_end'] = obj.span.end_sent + sent_start
                            sample['obj_type'] = obj.label
                            sample['token'] = tokens
                            sample['sent_start'] = sent_start
                            sample['sent_end'] = sent_end

                            sent_samples.append(sample)
                    if task == 'biored':
                        if type_pair in biored_entity_pairs: # This task is only for BioRED
                            if [sub.span.text, obj.span.text] not in captured:
                                label = gold_rel.get((sub.span, obj.span), 'no_relation')
                                if label == 'no_relation':
                                    label = gold_rel.get((obj.span, sub.span), 'no_relation')
                                sample = {}
                                sample['docid'] = doc._doc_key
                                sample['id'] = '%s@%d::(%d,%d)-(%d,%d)' % (
                                    doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc,
                                    obj.span.start_doc,
                                    obj.span.end_doc)
                                sample['relation'] = label
                                sample['subj_start'] = sub.span.start_sent + sent_start
                                sample['subj_end'] = sub.span.end_sent + sent_start
                                sample['subj_type'] = sub.label
                                sample['obj_start'] = obj.span.start_sent + sent_start
                                sample['obj_end'] = obj.span.end_sent + sent_start
                                sample['obj_type'] = obj.label
                                sample['token'] = tokens
                                sample['sent_start'] = sent_start
                                sample['sent_end'] = sent_end

                                sent_samples.append(sample)

                                captured.append([sub.span.text, obj.span.text])
                                captured.append([obj.span.text, sub.span.text])

                    if task == 'ddi':
                        # if type_pair in biored_entity_pairs: # This task is only for BioRED
                        if [sub.span.text, obj.span.text] not in captured:
                            label = gold_rel.get((sub.span, obj.span), 'no_relation')
                            if label == 'no_relation':
                                label = gold_rel.get((obj.span, sub.span), 'no_relation')
                                if label == 'no_relation':
                                    num_null += 1
                                    if num_null > 15000:
                                        continue
                            sample = {}
                            sample['docid'] = doc._doc_key
                            sample['id'] = '%s@%d::(%d,%d)-(%d,%d)' % (
                                doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc,
                                obj.span.start_doc,
                                obj.span.end_doc)
                            sample['relation'] = label
                            sample['subj_start'] = sub.span.start_sent + sent_start
                            sample['subj_end'] = sub.span.end_sent + sent_start
                            sample['subj_type'] = sub.label
                            sample['obj_start'] = obj.span.start_sent + sent_start
                            sample['obj_end'] = obj.span.end_sent + sent_start
                            sample['obj_type'] = obj.label
                            sample['token'] = tokens
                            sample['sent_start'] = sent_start
                            sample['sent_end'] = sent_end

                            sent_samples.append(sample)

                            captured.append([sub.span.text, obj.span.text])
                            captured.append([obj.span.text, sub.span.text])

                    if task == 'scierc':
                        if [sub.span, obj.span] not in captured:
                            label = gold_rel.get((sub.span, obj.span), 'no_relation')
                            rev_label = ''
                            # for symmetric relation labels, we only train and eval these instances once
                            if label == 'no_relation':
                                rev_label = gold_rel.get((obj.span, sub.span), 'no_relation')
                            if label in ['CONJUNCTION', 'COMPARE'] or rev_label in ['CONJUNCTION', 'COMPARE']:
                                captured.append([sub.span, obj.span])
                                captured.append([obj.span, sub.span])
                                if rev_label in ['CONJUNCTION', 'COMPARE']:
                                    label = rev_label
                            sample = {}
                            sample['docid'] = doc._doc_key
                            sample['id'] = '%s@%d::(%d,%d)-(%d,%d)' % (
                                doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc,
                                obj.span.end_doc)
                            sample['relation'] = label
                            sample['subj_start'] = sub.span.start_sent + sent_start
                            sample['subj_end'] = sub.span.end_sent + sent_start
                            sample['subj_type'] = sub.label
                            sample['obj_start'] = obj.span.start_sent + sent_start
                            sample['obj_end'] = obj.span.end_sent + sent_start
                            sample['obj_type'] = obj.label
                            sample['token'] = tokens
                            sample['sent_start'] = sent_start
                            sample['sent_end'] = sent_end

                            sent_samples.append(sample)
                    if task=='semeval' and [sub.span.text, obj.span.text] not in captured:
                        label = gold_rel.get((sub.span, obj.span), 'no_relation')
                        if label == 'Other':
                            label = 'no_relation'
                        if label == 'no_relation':
                            label = gold_rel.get((obj.span, sub.span), 'no_relation')
                        if label == 'Other':
                            label = 'no_relation'
                        sample = {}
                        sample['docid'] = doc._doc_key
                        sample['id'] = '%s@%d::(%d,%d)-(%d,%d)' % (
                            doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc,
                            obj.span.end_doc)
                        sample['relation'] = label
                        sample['subj_start'] = sub.span.start_sent + sent_start
                        sample['subj_end'] = sub.span.end_sent + sent_start
                        sample['subj_type'] = sub.label
                        sample['obj_start'] = obj.span.start_sent + sent_start
                        sample['obj_end'] = obj.span.end_sent + sent_start
                        sample['obj_type'] = obj.label
                        sample['token'] = tokens
                        sample['sent_start'] = sent_start
                        sample['sent_end'] = sent_end

                        sent_samples.append(sample)
                        captured.append([sub.span.text, obj.span.text])
                        captured.append([obj.span.text, sub.span.text])

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples

    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d' % (tot, max_sentsample))

    return data, samples, nrel