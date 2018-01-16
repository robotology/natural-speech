#include <yarp/sig/SoundFile.h>
 using namespace yarp::sig::file;
 #include <yarp/dev/all.h>
 using namespace yarp::dev;

 Sound s;
 bool ok = read(s,"source.wav");
 if (!ok) { printf("FAIL\n"); exit(1); }
 PolyDriver dd;
 Property config;
 config.put("device","portaudio");
 config.put("write","1");
 ... // may need to choose an output
 ok = dd.open(config);
 if (!ok) { printf("FAIL (device)\n"); exit(1); }
 IAudioRender *audio_out = NULL;
 ok = dd.view(audio_out);
 if (!ok) { printf("FAIL (interface)\n"); exit(1); }
 dd->renderSound(s);
