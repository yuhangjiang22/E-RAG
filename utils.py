import logging

from data_structures import Dataset

logger = logging.getLogger('root')

type_map = {
    'DiseaseOrPhenotypicFeature': 'disease',
    'SequenceVariant': 'variant',
    'GeneOrGeneProduct': 'gene',
    'ChemicalEntity': 'drug'
}


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
                    if task == 'cdr':
                        if sub.label == 'chemical' and obj.label == 'disease':
                            label = gold_rel.get((sub.span, obj.span), 'no_relation')
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

                    if task == 'biored':
                        if type_pair in biored_entity_pairs:
                            head = ' '.join(sub.span.text)
                            tail = ' '.join(obj.span.text)
                            head_type, tail_type = type_pair
                            if head + '$$' + head_type + '$$' + tail + '$$' + tail_type in captured:
                                continue
                            if tail + '$$' + tail_type + '$$' + head + '$$' + head_type in captured:
                                continue

                            label = gold_rel.get((sub.span, obj.span), 'no_relation')
                            reverse_label = gold_rel.get((obj.span, sub.span), 'no_relation')

                            if label != 'no_relation':
                                real_label = label
                            if reverse_label != 'no_relation':
                                real_label = reverse_label
                            if label == 'no_relation' and reverse_label == 'no_relation':
                                real_label = 'no_relation'

                            captured.append(head + '$$' + head_type + '$$' + tail + '$$' + tail_type)
                            captured.append(tail + '$$' + tail_type + '$$' + head + '$$' + head_type)

                            sample = {}
                            sample['docid'] = doc._doc_key
                            sample['id'] = '%s@%d::(%d,%d)-(%d,%d)' % (
                                doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc,
                                obj.span.start_doc,
                                obj.span.end_doc)
                            sample['relation'] = real_label
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