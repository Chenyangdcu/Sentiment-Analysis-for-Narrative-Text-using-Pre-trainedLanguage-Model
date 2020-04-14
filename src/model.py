# coding=utf-8
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss
import argparse
import sklearn.metrics as metric
import glob
import logging
import os
import random
import pandas as pd
import numpy as np
import json
from os import listdir
from os.path import isfile, join

from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import InputExample

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
# label_list = ["1", "2", "3", "4", "5"]
label_list = ["A", "D", "F", "H", "N", "Sa", "Su+", "Su-"]
binary_label_list = ["N", "E"]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
}


test_filenames = ['data/Grimms/emmood/the_turnip.emmood', 'data/Grimms/emmood/the_story_of_the_youth_who_went_forth_to_learn_what_fear_was.emmood', 'data/Grimms/emmood/22_the_riddle.emmood', 'data/Grimms/emmood/29_the_devil_with_the_three_golden_hairs.emmood', 'data/Grimms/emmood/the_golden_goose.emmood', 'data/Grimms/emmood/the_twelve_dancing_princesses.emmood', 'data/Grimms/emmood/73_the_wolf_and_the_fox.emmood', 'data/Grimms/emmood/briar_rose.emmood', 'data/Potter/emmood/the_roly-poly_pudding.emmood', 'data/HCAndersen/emmood/drop_wat.emmood', 'data/HCAndersen/emmood/lovelies.emmood', 'data/HCAndersen/emmood/last_dre.emmood', 'data/HCAndersen/emmood/bell.emmood', 'data/HCAndersen/emmood/races.emmood', 'data/HCAndersen/emmood/buckwhet.emmood', 'data/HCAndersen/emmood/heaven.emmood']


class Arguments:
    # used to store parameters
    def __init__(self, name):
        self.name = name
        self.max_seq_length = 128
        self.output_mode = "classification"
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.n_gpu = 1
        self.seed = 1
        self.do_train = False
        self.do_eval = True
        self.train_batch_size = 16
        self.eval_batch_size = 1
        self.num_train_epochs = 0
        self.weight_decay = 0.0
        self.learning_rate = 3e-5
        self.adam_epsilon = 1e-8
        self.warmup_steps = 500
        self.model_name_or_path = 'bert-base-cased'
        self.model_type = "bert"
        self.max_grad_norm = 1.0
        self.logging_steps = 400
        self.save_steps = 200000
        self.output_dir = "trained_model/new_binary_fairy_bert/"
        # self.output_dir = "bert_context_epoch2_lr8e-06_steps40000_ratio5"
        self.eval_all_checkpoints = False
        self.do_lower_case = False
        self.gradient_accumulation_steps = 1
        self.task_name = "sst-5"
        # self.eval_file = "data/sst/sst_test.txt"
        self.eval_file = "data/dr_dataset_adjust.txt"
        self.context = False
        self.knowledge = False
        self.knowledge_file = "data/knowledge_ibm/idiomLexicon.tsv"
        self.eval_out_file = "eval_results_fairy/new_binary_fairy_.json"
        self.split_ratio = 0.1
        self.k_ratio = 0.15
        self.is_binary = True


args = Arguments("fairy-tales")
model_name = 'bert-base-cased'


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, c_indices):
    """ Train the model """
    tb_writer = SummaryWriter()

    _label_list = label_list if args.is_binary is False else binary_label_list

    num_labels = len(_label_list)

    train_sampler = SequentialSampler(train_dataset) if args.context else RandomSampler(train_dataset)
    # train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1 if args.context else args.train_batch_size)

    # t_total = len(train_dataloader) // args.num_train_epochs
    t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # # Check if saved optimizer or scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #         os.path.join(args.model_name_or_path, "scheduler.pt")
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    #
    # # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    #
    # logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #
    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        iii = 0

        for step, batch in enumerate(epoch_iterator):
            iii += 1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM, DistilBERT and RoBERTa don't use segment_ids

            if args.context:
                inputs.pop("labels")
                outputs = model.bert(**inputs)
                pooled_output = outputs[1]

                pooled_output = model.dropout(pooled_output)

                if iii != 1:
                    if c_indices[iii-1] == c_indices[iii-2]:
                        pooled_output += contexts * args.context_ratio
                contexts = pooled_output.clone().detach()
                logits = model.classifier(pooled_output)
            else:
                outputs = model(**inputs)

                logits = outputs[1]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), batch[3].view(-1))

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    # if (
                    #     args.local_rank == -1 and args.evaluate_during_training
                    # ):  # Only evaluate when single GPU otherwise metrics may not average well
                    #     results = evaluate(args, model, tokenizer)
                    #     for key, value in results.items():
                    #         eval_key = "eval_{}".format(key)
                    #         logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    tb_writer.close()
    global_step = 1 if global_step == 0 else global_step
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_sets, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    _label_list = label_list if args.is_binary is False else binary_label_list

    softmax = torch.nn.Softmax(dim=1)

    if args.knowledge:
        knowledge_df = pd.read_csv(args.knowledge_file, sep="\t")
        length_knowledge = len(knowledge_df)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # eval_dataset, raw_sents, chapter_indices = load_data(args, args.eval_file, tokenizer)
        eval_dataset, raw_sents, c_indices = eval_sets[0], eval_sets[1], eval_sets[2]
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # args.eval_batch_size = args.train_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        sent_index = 0
        iii = 0
        chapter_sentis = {}
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            iii += 1
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                if args.context:
                    outputs = model.bert(**inputs)
                    if (iii - 1) == 172:
                        attn = outputs[2][11]
                        att = torch.zeros(1, 128, 128).to(args.device)

                        # for j in range(12):
                        #     att += attn[:, j, :]
                        att = attn[:, 2, :]
                        att = att.view(128, 128)
                        print(att[:25, :25])
                        # hidden_states = outputs[0].view(args.max_seq_length, -1)
                        # attn_weights = torch.mm(hidden_states, hidden_states.transpose(0, 1))
                        #
                        # print(softmax(attn_weights[:25, :25]))
                        return
                    pooled_output = outputs[1]

                    pooled_output = model.dropout(pooled_output)

                    if iii != 1:
                        if c_indices[iii - 1] == c_indices[iii - 2]:
                            pooled_output += contexts * args.context_ratio

                    contexts = pooled_output.clone().detach()
                    logits = softmax(model.classifier(pooled_output))

                    if args.knowledge:
                        for index in range(length_knowledge):
                            if knowledge_df.iloc[index][0] in raw_sents[sent_index]:
                                term_tensor = torch.tensor([knowledge_df.iloc[index][9], knowledge_df.iloc[index][7]+knowledge_df.iloc[index][8]], dtype=torch.float, device=args.device)
                                logits += (term_tensor*args.k_ratio)

                    inputs["labels"] = batch[3]
                else:
                    inputs["labels"] = batch[3]
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]
                    logits = softmax(logits)
                    if args.knowledge:
                        for index in range(length_knowledge):
                            if knowledge_df.iloc[index][0] in raw_sents[sent_index]:
                                term_tensor = torch.tensor([knowledge_df.iloc[index][9], knowledge_df.iloc[index][7]+knowledge_df.iloc[index][8]], dtype=torch.float, device=args.device)
                                logits += (term_tensor*args.k_ratio)

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            sent_index += 1

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        # for indi, p in enumerate(preds):
        #     if chapter_indices[indi] not in chapter_sentis:
        #         chapter_sentis[chapter_indices[indi]] = {"senti_score": p, "total_sentence": 1}
        #     else:
        #         chapter_sentis[chapter_indices[indi]]["senti_score"] += p
        #         chapter_sentis[chapter_indices[indi]]["total_sentence"] += 1
        result = {"acc": (preds == out_label_ids).mean()}
        results.update(result)

        eval_to_file(args.eval_out_file, preds, out_label_ids, chapter_sentis)
        print(result)
        print("accuracy: ", metric.accuracy_score(out_label_ids, preds))
        print("precision: ", metric.precision_score(out_label_ids, preds, average='macro'))
        print("recall: ", metric.recall_score(out_label_ids, preds, average='macro'))
        print("F1: ", metric.f1_score(out_label_ids, preds, average='macro'))

    return results


def soft_eval(preds, true_labels):
    acc = 0
    for p, l_ in zip(preds, true_labels):
        if p == l_:
            acc += 1
            continue
        if ((p - l_) == 1 or (p - l_) == -1) and p != 3 and l_ != 3:
            acc += 1
    return acc / len(preds)


def eval_to_file(out_file_name, preds, true_labels, chap_json):
    accuracy = metric.accuracy_score(true_labels, preds)
    precision = metric.precision_score(true_labels, preds, average='macro')
    recall = metric.recall_score(true_labels, preds, average='macro')
    f1 = metric.f1_score(true_labels, preds, average='macro')
    # _soft_eval = soft_eval(preds, true_labels)

    with open(out_file_name, "w") as f:
        line = "accuracy: " + str(accuracy) + "\n" + "precision: " + str(precision) + "\n" + \
               "recall: " + str(recall) + "\n" + "F1: " + str(f1) + "\n"

        chap_json["statistics"] = line

        json.dump(chap_json, f)

        for (p, t) in zip(preds, true_labels):
            line = str(p) + "---" + str(t) + "\n"
            f.write(line)


def load_data(args, data_dir, tokenizer):
    with open(data_dir, "r") as f:
        lines = f.readlines()
        examples = []
        raw_sents = []
        chapter_indices = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("SST-5", i)
            a, b = line.split('\t')
            text_a = b[:-1]
            label = line[9]
            chapter_index = a[10:]
            chapter_indices.append(chapter_index)
            raw_sents.append(text_a)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=args.output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if args.output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif args.output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset, raw_sents, chapter_indices


def load_data_tales(args, tokenizer):

    paths = ["data/Grimms/emmood/", "data/Potter/emmood/", "data/HCAndersen/emmood/"]

    _label_list = label_list if args.is_binary is False else binary_label_list

    total_filenames = []
    for p in paths:
        onlyfiles = [p+f for f in listdir(p) if isfile(join(p, f))]

        total_filenames += onlyfiles

    examples = []
    raw_sents = []
    chapter_indices = []
    train_labels = []

    c_index = 0
    for data_dir in total_filenames:
        if data_dir in test_filenames:
            continue
        with open(data_dir, "r") as f:
            lines = f.readlines()

            for (i, line) in enumerate(lines):
                guid = "%s-%s" % ("fairy_tales", i)
                a, b, c, d = line.split('\t')
                text_a = d[:-1]
                label = b.split(':')[0]

                train_labels.append(label)
                if args.is_binary:
                    label = "E" if label is not "N" else "N"


                chapter_indices.append(c_index)

                raw_sents.append(text_a)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            c_index += 1
    train_n_count = train_labels.count("N")/len(train_labels)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=_label_list,
        max_length=args.max_seq_length,
        output_mode=args.output_mode,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if args.output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif args.output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    test_examples = []
    test_raw_sents = []
    test_chapter_indices = []
    test_labels = []

    c_index = 0
    for data_dir in test_filenames:
        with open(data_dir, "r") as f:
            lines = f.readlines()

            for (i, line) in enumerate(lines):
                guid = "%s-%s" % ("fairy_tales", i)
                a, b, c, d = line.split('\t')
                text_a = d[:-1]
                label = b.split(':')[0]

                test_labels.append(label)
                if args.is_binary:
                    label = "E" if label is not "N" else "N"

                test_chapter_indices.append(c_index)

                test_raw_sents.append(text_a)
                test_examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            c_index += 1

    test_n_count = test_labels.count("N")/len(test_labels)
    features = convert_examples_to_features(
        test_examples,
        tokenizer,
        label_list=_label_list,
        max_length=args.max_seq_length,
        output_mode=args.output_mode,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if args.output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif args.output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    test_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return train_dataset, raw_sents, chapter_indices, test_dataset, test_raw_sents, test_chapter_indices


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--dir", type=str,  help="number of training epochs.")
    parser.add_argument("--epochs", type=int, default=1, help="number of training epochs.")
    parser.add_argument("--context", action="store_true", help="whether to incorporate contextual feature.")
    parser.add_argument("--learning_rate", type=float, default=0.0, help="learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="number of warmup steps for training.")
    parser.add_argument("--context_ratio", type=float, default=0.0, help="ratio of contextual feature.")
    #
    arguments = parser.parse_args()

    args.num_train_epochs = arguments.epochs
    args.context = arguments.context
    args.learning_rate = arguments.learning_rate
    args.context_ratio = arguments.context_ratio

    model_type = "context" if arguments.context else "vanilla"

    output_dir = "trained_model/" + "new_bert_" + model_type + "_" + "epoch" + str(arguments.epochs) + "_" + "lr" + str(arguments.learning_rate) + "_" + "steps" + str(arguments.warmup_steps) + "_" + "ratio" + str(int(arguments.context_ratio*10)) + "/"
    eval_dir = "eval_results_fairy/new_bert_" + model_type + "_" + "epoch" + str(arguments.epochs) + "_" + "lr" + str(arguments.learning_rate) + "_" + "steps" + str(arguments.warmup_steps) + "_" + "ratio" + str(int(arguments.context_ratio*10)) + ".json"

    args.output_dir = output_dir
    args.eval_out_file = eval_dir

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        do_lower_case=True,)
    train_dataset, train_raw_sents, train_c_index, test_dataset, test_raw_sents, test_c_index \
        = load_data_tales(args, tokenizer)

    eval_set = [test_dataset, test_raw_sents, test_c_index]

    # Set seed
    set_seed(args)

    _label_list = label_list if args.is_binary is False else binary_label_list

    num_labels = len(_label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    config.output_attentions = True
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        do_lower_case=True,
    )
    model = model_class.from_pretrained(
        model_name,
        config=config,
    )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, train_c_index)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # tokenizer = BertTokenizer.from_pretrained(model_name)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, eval_set, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
