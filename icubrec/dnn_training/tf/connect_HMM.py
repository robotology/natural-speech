#!/usr/bin/env python3

import argparse


def get_states(output_nodes_fname):
    file = open(output_nodes_fname, 'r')
    phonemes = [line.rstrip('\n') for line in file.readlines()]
    states = []
    for p in phonemes:
        for i in range(2, 5):
            label = p + '[' + str(i) + ']'
            states += [label]
    return states


def get_senones(output_nodes_fname):
    file = open(output_nodes_fname, 'r')
    senones = [line.rstrip('\n') for line in file.readlines()]
    return senones


def correct(original, lookup_table):
    corrected = []
    for l in original:
        if l.startswith("~h"):
            start = l.index('"') + 1
            end = l.index('"', start)
            phoneme = l[start:end]
        if l.startswith("<STATE>"):
            num_state = l[8:]
        if l.startswith("<TARGETINDEX>"):
            state = phoneme + "[" + num_state + "]"
            l = "<TARGETINDEX> " + lookup_table[state]
        corrected.append(l)
    return corrected


def correct_senones(original, lookup_table):
    corrected = []
    for l in original:
        if l.startswith("~s"):
            start = l.index('"') + 1
            end = l.index('"', start)
            senone_id = l[start:end]
        if l.startswith("<TARGETINDEX>"):
            l = "<TARGETINDEX> " + lookup_table[senone_id]
        corrected.append(l)
    return corrected


def connect(hmmdefs, output_nodes_fname, senones):
    if senones:
        output_nodes = get_senones(output_nodes_fname)
    else:
        output_nodes = get_states(output_nodes_fname)
    lookup_table = {}
    for s, i in zip(output_nodes, range(1, len(output_nodes) + 1)):
        lookup_table[s] = str(i)
    f = open(hmmdefs, 'r')
    lines = [l.rstrip("\n") for l in f.readlines()]
    f.close()
    if senones:
        lines = correct_senones(lines, lookup_table)
    else:
        lines = correct(lines, lookup_table)
    f = open(hmmdefs, 'w')
    f.write("\n".join(lines))
    f.close()


if __name__ == '__main__':
    # Parsing command line
    parser = argparse.ArgumentParser(
        description='Connect the output layer to the proper GMM output_nodes')
    parser.add_argument('hmmdefs', help='HMM defs file', type=str)
    parser.add_argument(
        'output_nodes_fname', metavar='phoneme_list',
        help='File containing the list of phonemes')
    parser.add_argument(
        '-s', '--senones', help='Connect to senones instead of output_nodes',
        dest='senones', action='store_true', default=False)
    args = parser.parse_args()
    connect(args.hmmdefs, args.output_nodes_fname, args.senones)
