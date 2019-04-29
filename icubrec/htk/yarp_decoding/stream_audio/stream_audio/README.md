
<a name='example-stream-audio'></a>
# Stream Audio

<p align='center'>
<img src="https://github.com/robotology/yarp.js/blob/master/images/example_browser_stream_audio.png" width="60%">
</p>

This example shows how to use Web APIs to send audio streams directly from any device microphone over the YARP network. In particular this application translates audio batches into [Yarp::sig::Sound](http://www.yarp.it/classyarp_1_1sig_1_1Sound.html) objects and then writes them on port `/yarpjs/mic:o`.

**Note.** For security reasons, Chrome does not allow to access the audio stream from unsecure hosts (https Vs http). To bypass this issue you have two options:

1. Use Firefox (which allows to access the audiostream also from unsecure domains)
2. Use a self-signed SSL scure domain (see [here](/examples#secure-domains) for a workaround).
