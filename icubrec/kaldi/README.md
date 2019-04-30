This folder contains the pipeline to perform online command recognition with
DNN-based command recognition. Pretrained models for the voice activity
detection (VAD) and the command recognition modules are not yet available but
will be added soon.

## Instruction to run the demo

* Ensure `yarpserver` is running
* Start `yarp-speech-sender` to stream the sound from the microphone of the
computer (output port: `/sender`) or the `sound_player` to play wav files (see
below for the second option).
* Start the VAD: `./pipeline/VAD_pipeline.py`
* Connect the input stream to the VAD (using one of the following commands,
depending on the chosen source, microphone or wav file)
```
yarp connect /sender /reader:i # microphone
yarp connnect /sound:o /reader:i # wav file
```
* Wait a few seconds and start the command recognition module
`./pipeline/CommandsRecognition_pipeline.py`
* The output of the command recognition system is available from port
`/cmd_writer:o`

After terminating the execution of the pipeline, temporary files can be
cleaned by running `./pipeline/save_cleaner.sh`.

### Playing sound with the `sound_player` module

* Run the `soundplayer` module: `../htk/build/bin/sound_player`
* Create an RPC client: `yarp rpc --client /rpc`
* In a new terminal, connect the RPC client to the `sound_player` RPC port:
`yarp connect /rpc /cmd:io`
* In the terminal where the RPC client is running, use the `play` command to
start streaming a wav file: `play <wav_file_path>`

## Content of the `/save` folder files

The pipelines generates the following files for each speech segment:

* `<id>.wav`: the audio segment selected by the VAD
* `<id>.scp`
* `<id>.post`
* `<id>.feat`: the features computed from the audio file and used as input to
the acoustic model
* `<id>.txt`: the output of the command recognizer
