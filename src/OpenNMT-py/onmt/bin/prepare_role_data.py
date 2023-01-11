#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs

import torch
from torch import nn
import csv
import os
import json
from collections import OrderedDict
import traceback
from tqdm import tqdm

import onmt.model_builder
import onmt.decoders.ensemble
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def gen_embds(opt, out_file=None):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    gpu_id = opt.gpu
    use_cuda = opt.gpu > -1
    device_id = torch.device("cuda", gpu_id) \
        if use_cuda else torch.device("cpu")

    fields, model, model_opt = load_model(opt)

    print(model.encoder)
    print(model.generator)
    print(model.generator._modules['0'])
    print(model.decoder)

    # export model weights
    export_model_weights(model, save_dir=opt.model_save_dir,
                         model_prefix=opt.model_prefix,
                         rnn_type=opt.rnn_type,
                         n_layers=opt.layers,
                         debug=opt.debug)

    src_shards = split_corpus(opt.src, opt.shard_size)
    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)

    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    tgt_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    shard_pairs = zip(src_shards, tgt_shards)

    with open(opt.output, 'w+') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        for shard_num, (src_shard, tgt_shard) in enumerate(shard_pairs):
            data_iter = preprocess_data_shard(opt, fields, device_id,
                                              src_reader, src_shard,
                                              tgt_reader, tgt_shard)

            for batch_num, batch in tqdm(enumerate(data_iter)):
                src, enc_states, memory_bank, src_lengths = run_encoder(batch, model)
                save_embd(src_shard, tgt_shard,
                          memory_bank, src_lengths, batch_num,
                          csv_writer)

        return


def save_embd(src_shard, tgt_shard,
              memory_bank, src_lengths, batch_num,
              csv_writer):
    batch_size = memory_bank.size()[1]
    start_idx = 1 * batch_num * batch_size
    end_idx = min(len(src_shard), start_idx + batch_size)

    src_text_input = src_shard[start_idx: end_idx]
    tgt_text_input = tgt_shard[start_idx: end_idx]

    text_input = zip(src_text_input, tgt_text_input)
    for idx, (src_text_seg, tgt_text_seg) in enumerate(text_input):
        mask_length = src_lengths[idx] - 1
        embd = memory_bank[mask_length, idx, :]

        fmt_embd = embd.cpu().detach().numpy().tolist()
        fmt_embd = [str(x) for x in fmt_embd]
        fmt_embd = " ".join(fmt_embd)

        src_text_seg = src_text_seg.decode('utf-8').strip("\n")
        tgt_text_seg = tgt_text_seg.decode('utf-8').strip("\n")

        # csv_writer.writerow([src_text_seg, tgt_text_seg, fmt_embd])
        csv_writer.writerow([src_text_seg, fmt_embd])

    return


def preprocess_data_shard(opt, fields, device_id,
                          src_reader, src_shard,
                          tgt_reader, tgt_shard):
    src_feats = {}
    src_data = {"reader": src_reader, "data": src_shard, "features": src_feats}

    tgt_feats = {}
    tgt_data = {"reader": tgt_reader, "data": tgt_shard, "features": tgt_feats}

    _readers, _data = inputters.Dataset.config(
        [('src', src_data), ('tgt', tgt_data)])

    filter_pred = None

    data = inputters.Dataset(
        fields,
        readers=_readers,
        data=_data,
        sort_key=inputters.str2sortkey[opt.data_type],
        filter_pred=filter_pred
    )

    data_iter = inputters.OrderedIterator(
        dataset=data,
        device=device_id,
        batch_size=opt.batch_size,
        batch_size_fn=max_tok_len if opt.batch_type == "tokens" else None,
        train=False,
        sort=False,
        sort_within_batch=False,
        shuffle=False
    )

    return data_iter


def load_model(opt):
    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )
    fields, model, model_opt = load_test_model(opt)

    vocabs = {'src': fields['src'].fields[0][-1], 'tgt': fields['tgt'].fields[0][-1], 'indices': fields['indices']}

    src_vocab = dict(vocabs['src'].vocab.stoi)
    tgt_vocab = dict(vocabs['tgt'].vocab.stoi)

    save_dir = opt.model_save_dir
    num_examples = f"{opt.model_prefix[0]}_example"

    src_vocab_savepath = os.path.join(save_dir, f"{num_examples}_src_vocab.json")
    tgt_vocab_savepath = os.path.join(save_dir, f"{num_examples}_tgt_vocab.json")

    with open(src_vocab_savepath, 'w+') as src_json_fh:
        json.dump(src_vocab, src_json_fh)
        print(f"SAVED src vocab at {src_vocab_savepath}")

    with open(tgt_vocab_savepath, 'w+') as tgt_json_fh:
        json.dump(tgt_vocab, tgt_json_fh)
        print(f"SAVED tgt vocab at {tgt_vocab_savepath}")

    return fields, model, model_opt


def run_encoder(batch, model):
    src, src_lengths = (
        batch.src if isinstance(batch.src, tuple) else (batch.src, None)
    )

    enc_states, memory_bank, src_lengths = model.encoder(
        src, src_lengths
    )
    if src_lengths is None:
        assert not isinstance(
            memory_bank, tuple
        ), "Ensemble decoding only supported for text data"
        src_lengths = (
            torch.Tensor(batch.batch_size)
            .type_as(memory_bank)
            .long()
            .fill_(memory_bank.size(0))
        )
    return src, enc_states, memory_bank, src_lengths


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements


def modify_ckpt_signature(model):
        model_state_dict = model.state_dict()

        modified_state_dict = OrderedDict()
        modified_state_dict['embedding.weight'] = \
            model_state_dict.pop('embedding.make_embedding.emb_luts.0.weight')

        for key, value in model_state_dict.items():
            if key.startswith('rnn') and 'layer' in key: # older version of pytorch. not compatible with A100 GPU
            # if key.startswith('rnn') and (key.endswith('l0') or key.endswith('l1')):
                layer_num = key.split(".")[2]
                # layer_num = key.rsplit("_")[-1]
                new_key = key.replace(f'layers.{layer_num}.', '')
                new_key = f"{new_key}_l{layer_num}"

                modified_state_dict[new_key] = model_state_dict[key]
            else:
                modified_state_dict[key] = model_state_dict[key]

        return modified_state_dict


def export_model_weights(model, save_dir=None,
                         model_prefix=None,
                         rnn_type='LSTM',
                         n_layers=1,
                         debug=False):
    cogs_decoder = torch.nn.Sequential(OrderedDict([
                                       ('embedding', model.decoder.embeddings),
                                       ('dropout', torch.nn.Dropout(p=0.0)),
                                       ('rnn', model.decoder.rnn),
                                       ('out', model.generator._modules['0']),
                                       ('softmax', model.generator._modules['2'])
                                        ]
                                        ))
    cogs_encoder =  torch.nn.Sequential(OrderedDict([
                                       ('embedding', model.encoder.embeddings),
                                       ('dropout', torch.nn.Dropout(p=0.0)),
                                       ('rnn', model.encoder.rnn),
                                        ]
                                        ))
    modified_encoder_state_dict = modify_ckpt_signature(cogs_encoder)
    modified_decoder_state_dict = modify_ckpt_signature(cogs_decoder)
    # modified_state_dict = cogs_decoder.state_dict()

    test_encoder = EncoderRNN(mytype=rnn_type, n_layers=n_layers)
    test_decoder = DecoderRNN(mytype=rnn_type, n_layers=n_layers)

    if debug:
        print("**"*50)
        print("Original Encoder State Dict\n")
        print(test_encoder.state_dict().keys())
        print("Modified Encoder State Dict\n")
        print(modified_encoder_state_dict.keys())
        print("**"*50)

        print("Original Decoder State Dict\n")
        print(test_decoder.state_dict().keys())
        print("**"*50)
        print("Modified Decoder State Dict\n")
        print(modified_decoder_state_dict.keys())
        print("**"*50)

    try:
        test_encoder.load_state_dict(modified_encoder_state_dict)
        print("\nSUCCESSFULLY LOADED MODIFIED ENCODER STATE DICT IN TEST ENCODER!\n")

    except Exception as err:
        print("\nFAILED TO LOAD MODIFIED ENCODER STATE DICT IN TEST ENCODER!\n")
        print(traceback.format_exc())
        raise err

    try:
        test_decoder.load_state_dict(modified_decoder_state_dict)
        print("\nSUCCESSFULLY LOADED MODIFIED DECODER STATE DICT IN TEST DECODER!\n")

    except Exception as err:
        print("\nFAILED TO LOAD MODIFIED DECODER STATE DICT IN TEST DECODER!\n")
        print(traceback.format_exc())
        raise err

    state = {'encoder_state_dict': modified_encoder_state_dict,
             'decoder_state_dict': modified_decoder_state_dict}
    if not os.path.exists(os.path.join(save_dir, f"cogs_ckpt_{model_prefix}.pt")):
        torch.save(state, os.path.join(save_dir, f"cogs_ckpt_{model_prefix}.pt"))
        print(f"SAVED MODEL WEIGHTS AT {os.path.join(save_dir, f'cogs_ckpt_{model_prefix}.pt')}")
    else:
        print(f'CHECKPOINT ALREADY EXISTS AT {os.path.join(save_dir, f"cogs_ckpt_{model_prefix}.pt")}')


class EncoderRNN(nn.Module):
    # def __init__(self, input_size=745, hidden_size=512, mytype='LSTM', n_layers=1, dropout_p=0.1):
    def __init__(self, input_size=675, hidden_size=512, mytype='LSTM', n_layers=1, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        # input_size is number of words in input vocab.
        self.type = mytype
        if self.type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        elif self.type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        elif self.type == 'SRN':
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        else:
            raise Exception('invalid network type')

    def forward(self, input_sequence, hidden=None):
        # Input
        #  input: vector of word indces
        #  hidden: (1 x 1 x hidden_size)
        #  NOTE: we should re-write to process batch of sequences,
        #   rather than individual words one by one
        seq_len = len(input_sequence)
        input_sequence = input_sequence.view(-1)
        embedded = self.embedding(input_sequence).view(seq_len, 1, -1)
        embedded = self.dropout(embedded)
        # embedded is (seq_len x 1 x hidden_size)
        output, hidden = self.rnn(embedded, hidden)
        # output is variable (seq_len x 1 x hidden_size)
        # hidden is variable (nlayer x 1 x hidden_size)
        return output, hidden

    def initHidden(self):
        # initialize variable of zeros the size of hidden state
        if self.type == 'LSTM':
            myhidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
            mycell = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
            if use_cuda:
                myhidden = myhidden.cuda()
                mycell = mycell.cuda()
            result = (myhidden, mycell)
        else:
            result = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
            if use_cuda:
                result = result.cuda()
        return result


class DecoderRNN(nn.Module):
    # def __init__(self, hidden_size=512, output_size=687, mytype='LSTM', n_layers=1, dropout_p=0.0):
    def __init__(self, hidden_size=512, output_size=655, mytype='LSTM', n_layers=1, dropout_p=0.0):
        # output_size : size of the output language vocabulary
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
                # output_size is number of words/actions in output vocab.
        self.dropout = nn.Dropout(dropout_p)
            # input_size is number of words in input vocab.
        self.type = mytype
        if self.type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        elif self.type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        else:
            raise Exception('invalid network type')

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()


    def forward(self, input, hidden):
        # Input
        #  input: a single word index (int)
        #  hidden : (nlayer x 1 x hidden_size)
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
            # embedded is (1 x 1 x hidden_size)
        # embedded = F.relu(embedded)  # Do we need this RELU?
        output, hidden = self.rnn(embedded, hidden)
            # output is variable (1 x 1 x hidden_size)
            # hidden is variable (nlayer x 1 x hidden_size)
            #   The last layer in hidden is the same as output, such that torch.equal(hidden[-1].data,output[0].data))
        netinput = self.out(output[0])
        output = self.softmax(netinput)
        # output is (1 x output_size), which is size of output language vocab
        return output, hidden, netinput


def _get_parser():
    parser = ArgumentParser(description='prepare_role_data.py')

    opts.config_opts(parser)
    opts.prepare_role_data_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    gen_embds(opt)


if __name__ == "__main__":
    main()
