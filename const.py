task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'chemprot': ['CHEMICAL', 'GENE'],
    'chemprot_5': ['CHEMICAL', 'GENE'],
    'biored': ['DiseaseOrPhenotypicFeature', 'SequenceVariant', 'GeneOrGeneProduct', 'ChemicalEntity'],
    'semeval': ['obj', 'sub'],
    'ddi': ["DRUG", "BRAND", "GROUP", "DRUG_N"],
    'cdr': ['disease', 'chemical']
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'chemprot': ['UPREGULATOR', 'ACTIVATOR', 'INDIRECT-UPREGULATOR',
                 'DOWNREGULATOR', 'INHIBITOR', 'INDIRECT-DOWNREGULATOR',
                 'AGONIST', 'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR',
                 'ANTAGONIST',
                 'SUBSTRATE', 'PRODUCT-OF', 'SUBSTRATE_PRODUCT-OF'],
    'chemprot_5': ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9'],
    'biored': ['Positive_Correlation', 'Negative_Correlation', 'Association', 'Bind', 'Drug_Interaction', 'Cotreatment', 'Comparison', 'Conversion'],
    'semeval': ['Cause-Effect', 'Instrument-Agency', 'Product-Producer', 'Content-Container', 'Entity-Origin', 'Entity-Destination',
                'Component-Whole', 'Member-Collection', 'Message-Topic'],
    'ddi': ['EFFECT', 'INT', 'MECHANISM', 'ADVISE'],
    'cdr': ['Inducement']
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label