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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EncoderRNN(nn.Module):
    def __init__(
        self, input_size=745, hidden_size=512, mytype="LSTM", n_layers=1, dropout_p=0.1
    ):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        # input_size is number of words in input vocab.
        self.type = mytype
        if self.type == "GRU":
            self.rnn = nn.GRU(
                hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p
            )
        elif self.type == "LSTM":
            self.rnn = nn.LSTM(
                hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p
            )
        else:
            raise Exception("invalid network type")

    def forward(self, input_sequence, hidden=None):
        # Input
        #  input: vector of word indces
        #  hidden: (1 x 1 x hidden_size)
        #  NOTE: we should re-write to process batch of sequences,
        #   rather than individual words one by one
        seq_len = input_sequence.size(1)
        # input_sequence = input_sequence.view(-1)

        embedded = self.embedding(input_sequence).view(seq_len, 1, -1)

        # print("embedded", embedded.shape)
        embedded = self.dropout(embedded)

        # embedded is (seq_len x 1 x hidden_size)
        output, hidden = self.rnn(embedded, hidden)
        # output is variable (seq_len x 1 x hidden_size)
        # hidden is variable (nlayer x 1 x hidden_size)
        return output, hidden

    def initHidden(self):
        # initialize variable of zeros the size of hidden state
        if self.type == "LSTM":
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
    # def __init__(self, hidden_size=512, output_size=687, mytype='LSTM', n_layers=1, dropout_p=0.1):
    def __init__(
        self, hidden_size=512, output_size=723, mytype="LSTM", n_layers=1, dropout_p=0.1
    ):
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
        if self.type == "GRU":
            self.rnn = nn.GRU(
                hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p
            )
        elif self.type == "LSTM":
            self.rnn = nn.LSTM(
                hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p
            )
        elif self.type == "SRN":
            self.rnn = nn.RNN(
                hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p
            )
        else:
            raise Exception("invalid network type")

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

        # print(hidden[0].size(), hidden[1].size())
        output, hidden = self.rnn(embedded, hidden)
        # output is variable (1 x 1 x hidden_size)
        # hidden is variable (nlayer x 1 x hidden_size)
        #   The last layer in hidden is the same as output, such that torch.equal(hidden[-1].data,output[0].data))
        netinput = self.out(output[0])
        output = self.softmax(netinput)
        # output is (1 x output_size), which is size of output language vocab
        return output, hidden, netinput


class AttnDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        output_size=687,
        mytype="LSTM",
        n_layers=1,
        dropout_p=0.1,
        attn_model="dot",
    ):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.attn_model = attn_model

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.type = mytype
        if self.type == "GRU":
            self.rnn = nn.GRU(
                self.hidden_size,
                self.hidden_size,
                num_layers=self.n_layers,
                dropout=self.dropout_p,
            )
        elif self.type == "LSTM":
            self.rnn = nn.LSTM(
                self.hidden_size,
                self.hidden_size,
                num_layers=self.n_layers,
                dropout=self.dropout_p,
            )
        else:
            raise Exception("invalid network type")

        self.attn = Attn(self.attn_model, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, prev_context, hidden, encoder_outputs):
        # runs for one timestep at a time

        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)

        # prev_context: (B, hidden_size)
        # input feeding, concat attentional vector with next input
        # dec_input = torch.cat((embedded, prev_context.unsqueeze(0)), 2)
        dec_input = embedded
        dec_outputs, hidden = self.rnn(dec_input, hidden)

        # encoder outputs: T, B, D * hidden_size
        # dec_outputs: T=1, B, D * hidden_size

        batch, source_l, dim = encoder_outputs.transpose(0, 1).size()
        batch_, target_l, dim_ = dec_outputs.transpose(0, 1).size()

        align_vectors = self.attn(
            dec_outputs.transpose(0, 1).contiguous(), encoder_outputs.transpose(0, 1)
        )

        # print(align_vectors.size(), encoder_outputs.transpose(0, 1).size())
        context = torch.bmm(align_vectors, encoder_outputs.transpose(0, 1))  # B x S x N

        concat_c = torch.cat(
            [context, dec_outputs.transpose(0, 1).contiguous()], 2
        ).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)

        attn_h = F.tanh(attn_h)

        # one step input
        attn_h = attn_h.squeeze(1)
        attn_weights = align_vectors.squeeze(1)

        output = self.out(attn_h)

        return output, hidden, context, attn_weights


class Attn(nn.Module):
    def __init__(self, method="dot", hidden_size=512):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        batch, source_l, dim = encoder_outputs.size()
        batch_, target_l, dim_ = hidden.size()

        align = self.score(hidden, encoder_outputs)

        align_vectors = F.softmax(align.view(batch * target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        return align_vectors

    def score(self, hidden, encoder_outputs):
        if self.method == "dot":
            encoder_outputs_ = encoder_outputs.transpose(1, 2)
            score = torch.bmm(hidden, encoder_outputs_)

        elif self.method == "general":
            length, batch_size, hidden_size = encoder_outputs.size()
            score = self.attn(encoder_outputs.view(-1, hidden_size)).view(
                length, batch_size, hidden_size
            )
            score = torch.bmm(hidden.transpose(0, 1), score.permute(1, 2, 0)).squeeze(1)

        elif self.method == "concat":
            length, batch_size, hidden_size = encoder_outputs.size()

            attn_input = torch.cat(
                (hidden.repeat(length, 1, 1), encoder_outputs), dim=2
            )

            score = self.attn(attn_input.view(-1, 2 * hidden_size)).view(
                length, batch_size, hidden_size
            )
            score = torch.bmm(
                self.v.repeat(batch_size, 1, 1), score.permute(1, 2, 0)
            ).squeeze(1)

        return score


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


def preprocess_data_shard(
    opt, fields, device_id, src_reader, src_shard, tgt_reader, tgt_shard
):
    src_feats = {}
    src_data = {"reader": src_reader, "data": src_shard, "features": src_feats}

    tgt_feats = {}
    tgt_data = {"reader": tgt_reader, "data": tgt_shard, "features": tgt_feats}

    _readers, _data = inputters.Dataset.config([("src", src_data), ("tgt", tgt_data)])

    filter_pred = None

    data = inputters.Dataset(
        fields,
        readers=_readers,
        data=_data,
        sort_key=inputters.str2sortkey[opt.data_type],
        filter_pred=filter_pred,
    )

    data_iter = inputters.OrderedIterator(
        dataset=data,
        device=device_id,
        batch_size=opt.batch_size,
        batch_size_fn=max_tok_len if opt.batch_type == "tokens" else None,
        train=False,
        sort=False,
        sort_within_batch=False,
        shuffle=False,
    )

    return data_iter


def load_model(opt):
    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )
    fields, model, model_opt = load_test_model(opt)

    vocabs = {
        "src": fields["src"].fields[0][-1],
        "tgt": fields["tgt"].fields[0][-1],
        "indices": fields["indices"],
    }

    src_vocab = dict(vocabs["src"].vocab.stoi)
    tgt_vocab = dict(vocabs["tgt"].vocab.stoi)

    return fields, model, model_opt, src_vocab, tgt_vocab


def modify_ckpt_signature(model, attn_model=True):
    model_state_dict = model.state_dict()

    modified_state_dict = OrderedDict()
    modified_state_dict["embedding.weight"] = model_state_dict.pop(
        "embedding.make_embedding.emb_luts.0.weight"
    )

    for key, value in model_state_dict.items():
        if key.startswith("rnn") and "layer" in key:
            layer_num = key.split(".")[2]
            new_key = key.replace(f"layers.{layer_num}.", "")
            new_key = f"{new_key}_l{layer_num}"

            modified_state_dict[new_key] = model_state_dict[key]
        else:
            modified_state_dict[key] = model_state_dict[key]

    return modified_state_dict


def export_model_weights(
    model, model_prefix=None, rnn_type="LSTM", attn_model=True, n_layers=1, debug=False
):

    print(model)
    # asdasdas
    # re-init COGS model so that we can modify the keys of the state_dict to
    # match the new model
    cogs_encoder = torch.nn.Sequential(
        OrderedDict(
            [
                ("embedding", model.encoder.embeddings),
                ("dropout", torch.nn.Dropout(p=0.0)),
                ("rnn", model.encoder.rnn),
            ]
        )
    )
    if attn_model:
        cogs_decoder = torch.nn.Sequential(
            OrderedDict(
                [
                    ("embedding", model.decoder.embeddings),
                    ("dropout", torch.nn.Dropout(p=0.0)),
                    ("rnn", model.decoder.rnn),
                    ("linear_out", model.decoder.attn.linear_out),
                    ("out", model.generator._modules["0"]),
                    ("cast", model.generator._modules["1"]),
                    ("log_softmax", model.generator._modules["2"]),
                ]
            )
        )
    else:
        cogs_decoder = torch.nn.Sequential(
            OrderedDict(
                [
                    ("embedding", model.decoder.embeddings),
                    ("dropout", torch.nn.Dropout(p=0.0)),
                    ("rnn", model.decoder.rnn),
                    ("out", model.generator._modules["0"]),
                    ("cast", model.generator._modules["1"]),
                    ("log_softmax", model.generator._modules["2"]),
                ]
            )
        )
    modified_encoder_state_dict = modify_ckpt_signature(cogs_encoder)
    modified_decoder_state_dict = modify_ckpt_signature(cogs_decoder)

    test_encoder = EncoderRNN(mytype=rnn_type, n_layers=n_layers)

    if attn_model:
        test_decoder = AttnDecoderRNN(mytype=rnn_type, n_layers=n_layers)
    else:
        test_decoder = DecoderRNN(mytype=rnn_type, n_layers=n_layers)

    if debug:
        print("**" * 50)
        print("Original Encoder State Dict\n")
        print(test_encoder.state_dict().keys())

        print("Modified Encoder State Dict\n")
        print(modified_encoder_state_dict.keys())
        print("**" * 50)

        print("Original Decoder State Dict\n")
        print(test_decoder.state_dict().keys())
        print("**" * 50)

        print("Modified Decoder State Dict\n")
        print(modified_decoder_state_dict.keys())
        print("**" * 50)

        print("Original COGS State Dict\n")
        print(model.state_dict().keys())

        print("**" * 50)

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

    return test_encoder, test_decoder


def decode(
    batch,
    encoder,
    decoder,
    src_vocab,
    tgt_vocab,
    use_attn=False,
    debug=True,
    model=None,
):
    if debug:
        print("*" * 50)
        print("BATCH)")
        print(batch)
        print("*" * 50)

    src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)

    tgt, tgt_lengths = batch.src if isinstance(batch.tgt, tuple) else (batch.tgt, None)

    if debug:
        print(f"src={src}, {src.size()}")
        print(f"src_lengths={src_lengths}")
        print("*" * 50)

        print(f"tgt={src}, {tgt.size()}")
        print(f"tgt_lengths={tgt_lengths}")
        print("*" * 50)

    src_index_to_word = {v: k for k, v in src_vocab.items()}
    tgt_index_to_word = {v: k for k, v in tgt_vocab.items()}

    encoder_input = src.permute(1, 2, 0).squeeze(0)
    tgt_sequence = tgt.permute(1, 2, 0).squeeze(0)

    # print(f"src.size()={src.size()}, encoder_input={encoder_input.size()}")

    if debug:
        print(f"encoder_input={encoder_input}, encoder_input={encoder_input.size()}")
        print("*" * 50)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(encoder_input)

    if debug:
        print(
            f"encoder_output={encoder_outputs.size()}, encoder_hidden={encoder_hidden.size()}"
        )
        print("*" * 50)

    batch_size = tgt_sequence.size(0)
    max_target_length = tgt_sequence.size(1)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[2]])).to(device=device)  # SOS

    if use_attn:
        decoder_context = Variable(torch.zeros(1, 512)).to(device=device)
    else:
        decoder_context = None

    decoder_hidden = encoder_hidden

    if debug:
        print(f"USING ATTN: {use_attn}")
        print(f"decoder_input={decoder_input.size()}")
        print(f"decoder_context={decoder_context.size()}")
        print("*" * 50)

    # Run through decoder
    decoded_words = ["<s>"]
    for t in range(max_target_length):
        if use_attn:
            decoder_output, decoder_hidden, decoder_context, decoder_attention = (
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            )
        else:
            decoder_output, decoder_hidden, decoder_raw_scores = decoder(
                decoder_input, decoder_hidden
            )

        topv, topi = decoder_output.data.topk(2)
        ni = topi[0][0]
        if ni == 3:
            decoded_words.append("</s>")
            break
        else:
            decoded_words.append(tgt_index_to_word[ni.item()])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni.item()]])).to(device=device)

    ## OPENNMT
    enc_hidden, enc_outputs, src_lengths = model.encoder(src, src_lengths)

    op_decoder_input = Variable(torch.LongTensor([[2]])).to(device=device)  # SOS

    if use_attn:
        op_decoder_context = Variable(torch.zeros(1, 512)).to(device=device)
    else:
        op_decoder_context = None

    op_decoder_hidden = enc_hidden

    if debug:
        print(f"USING ATTN: {use_attn}")
        print(f"op_decoder_input={op_decoder_input.size()}")
        print(f"op_decoder_context={op_decoder_context.size()}")
        print("*" * 50)

    # Run through decoder
    op_decoded_words = ["<s>"]
    for t in range(max_target_length):
        if use_attn:
            (
                op_decoder_output,
                op_decoder_hidden,
                op_decoder_context,
                op_decoder_attention,
            ) = decoder(
                op_decoder_input, op_decoder_context, op_decoder_hidden, enc_outputs
            )
        else:
            op_decoder_output, op_decoder_hidden, op_decoder_raw_scores = decoder(
                op_decoder_input, op_decoder_hidden
            )

        topv, topi = op_decoder_output.data.topk(2)
        ni = topi[0][0]
        if ni == 3:
            op_decoded_words.append("</s>")
            break
        else:
            op_decoded_words.append(tgt_index_to_word[ni.item()])

        # Next input is chosen word
        op_decoder_input = Variable(torch.LongTensor([[ni.item()]])).to(device=device)

    pred_text = " ".join(decoded_words)

    op_pred_text = " ".join(op_decoded_words)

    tgt_text = " ".join(
        [
            tgt_index_to_word[char.item()]
            for char in tgt_sequence.squeeze(0).cpu().numpy()
        ]
    )
    src_text = " ".join(
        [
            src_index_to_word[char.item()]
            for char in encoder_input.squeeze(0).cpu().numpy()
        ]
    )

    return (
        pred_text,
        tgt_text,
        src_text,
        op_pred_text,
        torch.isclose(encoder_hidden, enc_hidden).all(),
        torch.isclose(encoder_outputs, enc_outputs).all(),
    )


def run_inference(opt, out_file=None, debug=True):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    gpu_id = opt.gpu
    use_cuda = opt.gpu > -1
    device_id = torch.device("cuda", gpu_id) if use_cuda else torch.device("cpu")

    fields, model, model_opt, src_vocab, tgt_vocab = load_model(opt)

    print(model)

    if debug:
        print("*" * 50)
        print(model.encoder)

        print("*" * 50)
        print(model.generator)

        print("*" * 50)
        print(model.generator._modules["0"])

        print("*" * 50)
        print(model.decoder)
        print("*" * 50)

    # export model weights
    test_encoder, test_decoder = export_model_weights(
        model,
        model_prefix=opt.model_prefix,
        rnn_type=opt.rnn_type,
        attn_model=opt.attn_model,
        n_layers=opt.layers,
        debug=opt.debug,
    )

    test_encoder.eval()
    test_decoder.eval()

    print("openNMT model", model.encoder.training)

    src_shards = split_corpus(opt.src, opt.shard_size)
    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)

    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    tgt_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    shard_pairs = zip(src_shards, tgt_shards)

    num_correct, total = 0, 0
    num_agreement = 0
    enc_out_ag, enc_hidden_ag = 0, 0
    for shard_num, (src_shard, tgt_shard) in enumerate(shard_pairs):
        data_iter = preprocess_data_shard(
            opt, fields, device_id, src_reader, src_shard, tgt_reader, tgt_shard
        )

        for batch_num, batch in tqdm(enumerate(data_iter)):
            pred_text, tgt_text, input_text, op_pred_text, hidden_ag, out_ag = decode(
                batch,
                test_encoder,
                test_decoder,
                src_vocab,
                tgt_vocab,
                use_attn=opt.attn_model,
                debug=opt.debug,
                model=model,
            )
            if debug and batch_num % 10 == 0:
                print("*" * 50)
                print(
                    f"pred_text:{pred_text}\ntgt_text:{tgt_text}\ninput_text={input_text}\nop_pred_text={op_pred_text}\nhidden_ag={hidden_ag}\noutput_ag={out_ag}"
                )
                print("*" * 50)

            num_correct += int(pred_text == tgt_text)
            num_agreement += int(pred_text == op_pred_text)
            enc_hidden_ag += hidden_ag
            enc_out_ag += out_ag
            total += 1

    print(f"acc={num_correct/total}")
    print(f"agreement={num_agreement/total}")
    print(f"enc_hidden_agreement={enc_hidden_ag/total}")
    print(f"enc_output_agreement={enc_out_ag/total}")


def _get_parser():
    parser = ArgumentParser(description="prepare_role_data.py")

    opts.config_opts(parser)
    opts.prepare_role_data_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    run_inference(opt)


if __name__ == "__main__":
    main()
