#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import yarp
from utils.IndexableQueue import IndexableQueue
import time
from CommandRecognizer.CR_wrapper import CR_wrapper
from threading import Thread
import os
from signal_handler import SignalHandler
import argparse

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save/")

file_buffer = IndexableQueue()

#CR parameters
recipe_folder = '/home/storage/projects/kaldi/egs/vochime/s5_1ch'
CR_path = os.path.join(recipe_folder, "decode_example_gmm.sh")
CR_path_DNN = os.path.join(recipe_folder, "decode_example_dnn.sh")
#CR_path="/home/storage/projects/kaldi/egs/chime4_vocub/s5_1ch/decode_example_gmm.sh"
#CR_path_DNN="/home/storage/projects/kaldi/egs/chime4_vocub/s5_1ch/decode_example_dnn.sh"
#FE_path="/home/storage/projects/kaldi/egs/chime4_vocub/s5_1ch/feat_ext.sh"
FE_path = os.path.join(recipe_folder, "lda_feat_ext.sh")
#FE_path="/home/storage/projects/kaldi/egs/chime4_vocub/s5_1ch/lda_feat_ext.sh"
output_folder=save_folder
input_folder=save_folder
scp_files_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scp_file/")

output_delay = 15
command_recognizer = None


class CR_pipeline_updater(yarp.BottleCallback):

    def __init__(self):
        yarp.BottleCallback.__init__(self)

    def onRead(self, *argv):
        bottle = argv[0]
        out = bottle.get(0).toString_c()
        file_buffer.put(out)
        return True

def RC_function():
    while not SignalHandler.should_stop:
        if file_buffer.qsize() > 0:
            current_file=(file_buffer.pull_last_n_frame(1))
            if current_file is not None:
                current_file=current_file[0]
                index= os.path.splitext(os.path.basename(current_file))[0]
                print current_file
                command_recognizer.run_CR_dnn(current_file,index)


if __name__ == '__main__':
    # Initializing parameters
    parser = argparse.ArgumentParser(description='Command Recognition System.')
    parser.add_argument('-r', '--rejection_threshold', type=float, default=2, help='rejection threshold for the commands based on the acoustic cost')
    args = parser.parse_args()

    yarp.Network.init()
    cmd_output_port = yarp.Port()
    cmd_output_port.open("/cmd_writer:o")
    command_recognizer = CR_wrapper(CR_path=CR_path_DNN,output_folder=output_folder,input_folder=input_folder,
                                    scp_files_folder=scp_files_folder,FE_path=FE_path, cmd_output_port=cmd_output_port,
                                    rejection_threshold=args.rejection_threshold)
    p = yarp.BufferedPortBottle()
    updater = CR_pipeline_updater()
    p.open("/file_reader:i")
    yarp.Network.connect("/file_writer:o", "/file_reader:i")
    p.useCallback(updater)

    output_timer = 0
    while not SignalHandler.should_stop:
        thread_CR = Thread(target=RC_function)
        thread_CR.start()
        time.sleep(1)
	if output_timer % output_delay == 0:
		print("file:", file_buffer.qsize())
        output_timer += 1

    p.close()
