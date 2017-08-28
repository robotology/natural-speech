---
layout: page
title: VoCub dataset
tagline: A dataset of Vocal Commands for iCub
---
{% include JB/setup %}

The dataset can be downloaded from [here](https://zenodo.org/record/834934)

## Rational

We have created a dataset specifically for ASR for iCub. Recording a dataset has two main advantages: (i) it allows to easily test the recognition system and reliably estimate its performance in real conditions, and (ii) it can be used to adapt the recognition system in order to reduce the training/testing mismatch problem. This motivated us to record examples of the commands we want to recognize resulting in the creation of the VoCub dataset.

## Characteristics

The recordings consist of spoken English commands addressed to iCub. There are 103 unique commands (see below for some examples or [Resources](#resources) section for the complete list), composed of 62 different words. The command length ranges from 1 to 13 words, with an average of ~5 words per sentence. 29 speakers were recorded, 16 males and 13 females, 28 of which are non-native English speakers. We finally obtained 118 recordings from each speaker: of the 103 unique commands, 88 were recorded once, and 15 twice (corresponding to sentences containing rare words). This leads to about 2 hours and 30 minutes of recording in total.

|10 examples of the commands used in the VoCub dataset|
|-|
|I will teach you a new object.|
|This is an octopus.|
|What is this?|
|Let me show you how to reach the car with your left arm.|
|Let me show you how to reach the turtle with your right arm.|
|There you go.|
|Grasp the ladybug.|
|Where is the car?|
|No, here it is. |
|See you soon.|

## Files organization
A split of the speakers into training, validation and test sets is proposed with 21, 4 and 4 speakers per set respectively. The files are organized with the following convention `setid/spkrid/spkrid_cond_recid.wav`, where:
* `setid` identifies the set: `tr` for training, `dt` for validation and `et` for testing.
* `spkrid` identifies the speaker: from `001` to `021` for training, `101` to `104` for validation and `201` to `204` for testing.
* `cond` identifies the condition (see below).
* `recid` identifies the record within the condition (starting from `1` and increasing).

## Recording conditions

The commands were recorded in two different conditions (see video below for an illustration), non-static (`cond` = 1) and static condition (`cond` = 2),  with an equal number of recorded utterances per condition.

**Illustration of VoCub dataset acquisition procedure**
<iframe width="560" height="315" src="https://www.youtube.com/embed/N-rrNQ0gnRY" frameborder="0" allowfullscreen></iframe>

In the static condition, the speaker sat in front of two screens where the sentences to read were displayed. In the non-static condition, the commands were provided to the subject verbally through a speech synthesis system, and the person had to repeat them while performing a secondary manual task. This secondary task was designed to be simple enough to not impede the utterance repetition task, while requiring people to move around the robot. The distance between the speaker and the microphone in this last condition ranges from 50cm to 3m.

We also registered a set of additional sentences for the testing group (same structure but different vocabulary, see [Resources](#resources) section) to test a recognition system for new commands not seen during training. The sentences consist of 20 new commands, pronounced by each speaker of the test set twice: once in non-static condition (`cond` = 3) and once in static condition (`cond` = 4).

## Resources<a name="resources"></a>

* [List of original commands](sent.txt)
* [Original grammar](gram.txt)
* [Additionnal sentences](sent_add.txt) (conditions 3 and 4)
* [Full grammar](gram_all.txt), including additional sentences
