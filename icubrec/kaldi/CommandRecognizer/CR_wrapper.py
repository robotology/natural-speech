import os
import subprocess
import numpy as np
from output_cleaner import output_cleaner_dnn
from output_cleaner import output_cleaner
from CommandRecognizer.write_on_output_port import write_on_output_port
from AMDNN import AMModule
import time
import yarp

model_post = AMModule()

#open yarp port to send output

class CR_wrapper:

    def __init__(self, CR_path, output_folder, input_folder, scp_files_folder, FE_path=None, cmd_output_port=None, rejection_threshold=2):
        self.CR_path = CR_path
        self.FE_path = FE_path
        self.output_folder = output_folder
        self.input_folder = input_folder
        self.scp_files_folder=scp_files_folder
        self.cmd_output_port=cmd_output_port
	self.rejection_threshold = rejection_threshold

    def run_RC(self, input_file, index):
        output_file_path = os.path.join(self.output_folder, os.path.splitext(input_file)[0] + ".txt")
        input_file_path = os.path.join(self.input_folder, input_file)

        scp_file_path = os.path.join(self.scp_files_folder, os.path.splitext(input_file)[0] + ".scp")
        #DEBUG_
        scp_file = open(scp_file_path, 'w')
        scp_file.write(index +" "+ input_file_path)
        scp_file.close()
        # create scp_file
        subprocess.call(self.CR_path + " " + scp_file_path + " " + output_file_path, shell=True)
        output_cleaner(output_file_path)
        print "write: ", output_file_path

    def run_CR_dnn(self,input_file,index):

        CR_start_time= time.time()
        feat_file_path = os.path.join(self.output_folder, os.path.splitext(input_file)[0] + ".feat")
        input_file_path = os.path.join(self.input_folder, input_file)
        # feat_file_path = os.path.join(self.output_folder, os.path.splitext(input_file)[0] + ".txt")

        scp_file_path = os.path.join(self.scp_files_folder, os.path.splitext(input_file)[0] + ".scp")
        # DEBUG_
        scp_file = open(scp_file_path, 'w')
        scp_file.write(index + " " + input_file_path)
        scp_file.close()
        print "write scp_file:  ", time.time() - CR_start_time, " seconds - ",index
        FE_start_time = time.time()
        #features extraction: input scp_file_path (wav link), output  feat_file_path (.feat)
        subprocess.call(self.FE_path + " " + scp_file_path + " " + feat_file_path + "> /dev/null 2>&1", shell=True)
        audio_feature=np.genfromtxt(feat_file_path, delimiter=' ')
        #print audio_feature.shape
        print "compute features:  ", time.time() - FE_start_time, " seconds - ",index
        LP_start_time = time.time()
        log_post_file_path = os.path.join(self.output_folder, os.path.splitext(input_file)[0] + ".post")
        #print self.output_folder
        # print os.path.splitext(input_file)[0]
        # print log_post_file_path

        bool_output=model_post.GetOutput(input=audio_feature,file_id=index,output_file_path=log_post_file_path) #TODO:deve salvare il file
        print "compute log_post:  ", time.time() - LP_start_time, " seconds - ",index
        output_file_path = os.path.join(self.output_folder, os.path.splitext(input_file)[0] + ".txt")
        latgen_file_path = os.path.join(self.output_folder, os.path.splitext(input_file)[0] + ".latgen")

        subprocess.call(self.CR_path + " " + log_post_file_path + " " + output_file_path + " " + latgen_file_path + "> /dev/null 2>&1", shell=True)
        if output_cleaner_dnn(output_file_path, latgen_file_path, threshold=self.rejection_threshold):
            write_on_output_port(output_file_path, self.cmd_output_port)
        print "CR takes ", time.time() - CR_start_time, " seconds - ",index
        print "write: ", output_file_path
        
if __name__ == '__main__':
    _FE_path = "/home/storage/projects/kaldi/egs/chime4_vocub/s5_1ch/feat_ext.sh"
    _CR_path = "/home/storage/projects/kaldi/egs/chime4_vocub/s5_1ch/decode_example_dnn.sh"
    _input_folder = "/home/storage/projects/Demo2Icub/pipeline/save_old/"
    _scp_file_folder = "/home/storage/projects/kaldi/egs/chime4_vocub/s5_1ch/exp/scp_input_files/"
    _input_file = "13989831_008_1_005.wav"
    _output_folder="./"
    _index="777_1_019"

    wrapper = CR_wrapper(CR_path=_CR_path,output_folder=_output_folder, input_folder=_input_folder,
                         scp_files_folder=_scp_file_folder, FE_path=_FE_path)
    wrapper.run_CR_dnn(_input_file,_index)
