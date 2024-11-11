class args:
    num_docs = 1
    document_path = 'chemprot/chemprot_entity_pair.json'
    max_seq_length = 512
    train_file = 'chemprot/train.json'
    context_window = 0
    task = 'chemprot_5'
    output_dir = 'output'
    negative_label = 'no_relation'
    add_new_tokens = True
    model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    do_lower_case = True
    eval_batch_size = 4