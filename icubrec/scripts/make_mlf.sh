#!/bin/bash
# Creates a word level MLF for all the files

# Cleanup old files
rm -f $1/prune.log $1/missing.log $1/missing.txt $1/dataset.scp $1/words.mlf

# Create a file listing all the MFC files in the training directory
if test -z $MFC_FILES; then
    rm -f $1/mfc_files.txt
    find $MFC_FOLDER -iname '*.mfc' >$1/mfc_files.txt
    MFC_FILES=$1/mfc_files.txt
fi

# Create a file that contains the filename of all the transcription files
if test -z $DOT_FILES; then
    rm -f $1/dot_files.txt
    find -L $TRANS_FOLDER -iname '*.dot' >$1/dot_files.txt
    DOT_FILES=$1/dot_files.txt
fi

# Now create the MLF file using a script, we prune out anything that
# has words that aren't in our dictionary, producing a MLF with only
# these files and a corresponding script file.
perl $HTK_SCRIPTS/$MLF_CREATION_SCRIPT $2 $MFC_FILES $DOT_FILES $HTK_DATA/cmu/cmu6 $1/words.mlf $1/dataset.scp 1 "" $1/missing.txt 1 >$1/missing.log
