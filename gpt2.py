# import argparse
import ingest
import csv
from tqdm import tqdm
from transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME,
                                     get_linear_schedule_with_warmup)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import transformers
import numpy as np
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OpenAIGPTDoubleHeadsModel.from_pretrained("log/")
tokenizer = OpenAIGPTTokenizer.from_pretrained("log/")
special_tokens = ['_start_', '_delimiter_', '_classify_']
special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def load_rocstories_dataset(dataset_path, loadLabel=False):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    with open(dataset_path, encoding='utf_8') as f:
        f = csv.reader(f)
        output = []
        next(f) # skip the first line
        for line in tqdm(f):
            if(loadLabel):
                output.append((' '.join(line[1:5]), line[5], line[6], int(line[-1])-1))
            else:
                output.append((' '.join(line[1:5]), line[5], line[6]))
    return output

def tokenize_and_encode(obj):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, int):
        return obj
    return list(tokenize_and_encode(o) for o in obj)

def pre_process_datasets(dataset, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)
        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    n_batch = len(dataset)
    input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
    mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
    lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
    mc_labels = np.zeros((n_batch,), dtype=np.int64)
    for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
        with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
        input_ids[i, 0, :len(with_cont1)] = with_cont1
        input_ids[i, 1, :len(with_cont2)] = with_cont2
        mc_token_ids[i, 0] = len(with_cont1) - 1
        mc_token_ids[i, 1] = len(with_cont2) - 1
        lm_labels[i, 0, :len(with_cont1)] = with_cont1
        lm_labels[i, 1, :len(with_cont2)] = with_cont2
        mc_labels[i] = mc_label
    all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
    return tuple(torch.tensor(t) for t in all_inputs)

def pre_process_test_datasets(dataset, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)
        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    n_batch = len(dataset)
    input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
    mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
    lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
    for i, (story, cont1, cont2), in enumerate(dataset):
        with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
        input_ids[i, 0, :len(with_cont1)] = with_cont1
        input_ids[i, 1, :len(with_cont2)] = with_cont2
        mc_token_ids[i, 0] = len(with_cont1) - 1
        mc_token_ids[i, 1] = len(with_cont2) - 1
        lm_labels[i, 0, :len(with_cont1)] = with_cont1
        lm_labels[i, 1, :len(with_cont2)] = with_cont2
    all_inputs = (input_ids, mc_token_ids, lm_labels)
    return tuple(torch.tensor(t) for t in all_inputs)


validData = load_rocstories_dataset("data/dev.csv", loadLabel=True)
testData = load_rocstories_dataset("data/test.csv")
datasets = (validData, testData)
encoded_datasets = tokenize_and_encode(datasets)

max_length = model.config.n_positions // 2 - 2
input_length = max(len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) + 3  \
                        for dataset in [encoded_datasets[0]] for story, cont1, cont2, _ in dataset)
input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model


eval_tensor_dataset = pre_process_datasets(encoded_datasets[0], input_length, max_length, *special_tokens_ids)
test_tensor_dataset = pre_process_test_datasets(encoded_datasets[1], input_length, max_length, *special_tokens_ids)


eval_data = TensorDataset(*eval_tensor_dataset)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=16)


test_data = TensorDataset(*test_tensor_dataset)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=16)

model.eval()
# eval_loss, eval_accuracy = 0, 0
# nb_eval_steps, nb_eval_examples = 0, 0
# for batch in tqdm(eval_dataloader, desc="Evaluating"):
#     batch = tuple(t.to(device) for t in batch)
#     input_ids, mc_token_ids, lm_labels, mc_labels = batch
#     with torch.no_grad():
#         _, mc_loss, _, mc_logits = model(input_ids, mc_token_ids=mc_token_ids, lm_labels=lm_labels, mc_labels=mc_labels)

#     mc_logits = mc_logits.detach().cpu().numpy()
#     mc_labels = mc_labels.to('cpu').numpy()
#     tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

#     eval_loss += mc_loss.mean().item()
#     eval_accuracy += tmp_eval_accuracy

#     nb_eval_examples += input_ids.size(0)
#     nb_eval_steps += 1

# eval_loss = eval_loss / nb_eval_steps
# eval_accuracy = eval_accuracy / nb_eval_examples
# result = {'eval_loss': eval_loss,
#             'eval_accuracy': eval_accuracy}

with open("gpt2/result.csv", "w") as f:
    f.write("Id,Prediction\n")
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, mc_token_ids, lm_labels,_ = batch
        with torch.no_grad():
            mc_loss, _, mc_logits = model(input_ids, mc_token_ids=mc_token_ids, lm_labels=lm_labels)
            for i in range(len(mc_logits)):
                logit = mc_logits[i]
                label = np.argmax(logit).item() + 1
                f.write(str(label))
        mc_logits = mc_logits.detach().cpu().numpy()
