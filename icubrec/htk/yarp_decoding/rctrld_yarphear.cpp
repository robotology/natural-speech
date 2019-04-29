/*
 * Copyright (C) 2006, 2007 RobotCub Consortium
 * Authors: Paul Fitzpatrick
 * CopyPolicy: Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
 *
 */


#include <deque>

#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>

#include <yarp/os/all.h>
#include <yarp/os/Log.h>

#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/AudioGrabberInterfaces.h>

#include <yarp/sig/SoundFile.h>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::file;
using namespace yarp::dev;


bool interrupt = false;

class Echo : public TypedReaderCallback<Sound>, public TypedReaderCallback<Bottle> {
public:
    ConstString fname;

private:
    PolyDriver poly;
    IAudioRender *put;
    BufferedPort<Sound> mic_port;
    BufferedPort<Bottle> cmd_port;
    BufferedPort<Bottle> log_port;
    Semaphore mutex;
    bool muted;
    bool saving;
    std::deque<Sound> sounds;
    int samples;
    int channels;
    int ct;
    int padding;

public:
    Echo() : mutex(1) {
        put = NULL;
        mic_port.useCallback(*this);
        mic_port.setStrict();
        cmd_port.useCallback(*this);
        muted = false;
        saving = false;
        samples = 0;
        channels = 0;
        put = NULL;
        ct = 0;
        padding = 0;
        fname = "audio_%06d.wav";
    }

    bool open(Searchable& p) {
        if (!cmd_port.open("/cmd:i")) {
            yError("Communication problem\n");
            return false;
        }
        if (!log_port.open("/log:o")) {
            yError("Communication problem\n");
            return false;
        }

        bool dev = true;
        if (p.check("nodevice")) {
            dev = false;
        }
        if (dev) {
            poly.open(p);
            if (!poly.isValid()) {
                yError("cannot open driver\n");
                return false;
            }

            if (!p.check("mute")) {
                // Make sure we can write sound
                poly.view(put);
                if (put==NULL) {
                    yError("cannot open interface\n");
                    return false;
                }
            }
        }

        mic_port.setStrict(true);
        if (!mic_port.open(p.check("name",Value("/mic:i")).asString())) {
            yError("Communication problem\n");
            return false;
        }

        if (p.check("remote")) {
            Network::connect(p.check("remote",Value("/remote")).asString(),
                             mic_port.getName());
        }

        return true;
    }

    using TypedReaderCallback<Sound>::onRead;
    void onRead(Sound& sound)
     {
        #ifdef TEST
        //this block can be used to measure time elapsed between two sound packets
        static double t1= yarp::os::Time::now();
        static double t2= yarp::os::Time::now();
        t1= yarp::os::Time::now();
        yDebug("onread %f\n", t2-t1);
        t2 = yarp::os::Time::now();
        #endif

        int ct = mic_port.getPendingReads();
        //yDebug("pending reads %d\n", ct);
        while (ct>padding) {
            ct = mic_port.getPendingReads();
            yWarning("Dropping sound packet -- %d packet(s) behind\n", ct);
            mic_port.read();
        }
        mutex.wait();

        if (!muted) {
            if (put!=NULL) {
                put->renderSound(sound);
            }
        }
        if (saving) {
            saveFrame(sound);
        }

        mutex.post();
        Time::yield();
    }

    using TypedReaderCallback<Bottle>::onRead;
    virtual void onRead(Bottle& command) {
        bool help = false;

        ConstString cmd = command.get(0).asString();
        if (command.size()==0) {
            mute(!muted);
            yInfo("%s\n", muted ? "Muted" : "Audible again");
        } else if (cmd=="mute") {
            mute(true);
            yInfo("Muted\n");
        } else if (cmd=="unmute") {
            mute(false);
            yInfo("Audible again\n");
        } else if (cmd=="help") {
            help = true;
        } else if (cmd=="s") {
            save(!saving);
            yInfo("%s\n", saving ? "Saving" : "Stopped saving");
            if (saving) {
                yInfo("  Type \"s\" again to stop saving\n");
            }
        } else if (cmd=="write"||cmd=="w") {
            if (command.size()==2) {
                fname = command.get(1).asString();
            }
            char buf[2560];
            sprintf(buf, fname.c_str(), ct);
            saveFile(buf);
            log("write", fname.c_str());
            ct++;
        } else if (cmd=="q"||cmd=="quit") {
            interrupt = true;
        } else if (cmd=="buf"||cmd=="b") {
            padding = command.get(1).asInt();
            yInfo("Buffering at %d\n", padding);
        }

        if (help) {
            yInfo("  Press return to mute/unmute, or ...\n");
            yInfo("  Type \"mute\" to mute\n");
            yInfo("  Type \"unmute\" to unmute\n");
            yInfo("  Type \"s\" to set start/stop saving audio in memory\n");
            yInfo("  Type \"w[rite]\" to write saved audio with same/default name\n");
            yInfo("  Type \"w[rite] filename.wav\" to write saved audio to a file\n");
            yInfo("  Type \"b[uf] NUMBER\" to set buffering delay (default is 0)\n");
            yInfo("  Type \"q\" to quit\n");
            yInfo("  Type \"help\" to see this list again\n");
            help = false;
        } else {
            yInfo("Type \"help\" for usage\n");
        }
    }

    void log(ConstString action, ConstString description="") {
        Bottle& log = log_port.prepare();
        log.clear();
        log.addString(action.c_str());
        if (description != "")
            log.addString(description.c_str());
        log_port.write();
    }

    void mute(bool muteFlag=true) {
        mutex.wait();
        muted = muteFlag;
        mutex.post();
    }

    void save(bool saveFlag=true) {
        mutex.wait();
        saving = saveFlag;
        mutex.post();
    }

    void saveFrame(Sound& sound) {
        sounds.push_back(sound);
        samples += sound.getSamples();
        channels = sound.getChannels();
        yDebug("  %ld sound frames buffered in memory (%ld samples)\n",
               (long int) sounds.size(),
               (long int) samples);
    }

    bool saveFile(const char *name) {
        mutex.wait();
        saving = false;

        Sound total;
        total.resize(samples,channels);
        long int at = 0;
        while (!sounds.empty()) {
            Sound& tmp = sounds.front();
            for (int i=0; i<channels; i++) {
                for (int j=0; j<tmp.getSamples(); j++) {
                    total.set(tmp.get(j,i),at+j,i);
                }
            }
            total.setFrequency(tmp.getFrequency());
            at += tmp.getSamples();
            sounds.pop_front();
        }
        mutex.post();
        bool ok = write(total,name);
        if (ok) {
            yDebug("Wrote audio to %s\n", name);
        }
        samples = 0;
        channels = 0;
        return ok;
    }

    bool getSaving(){
        return saving;
    }

    bool close() {
        mic_port.close();
        cmd_port.close();
        log_port.close();
        if (poly.isValid()) {
            poly.close();
        }
        mutex.wait(); // onRead never gets called again once it finishes
        return true;
    }
};

void sig_handler(int s){
    printf("Caught signal %d\n",s);
    interrupt = true;
}

int main(int argc, char *argv[]) {
    // Allowing to catch interruptions to close gracefully
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = sig_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    sigaction(SIGTERM, &sigIntHandler, NULL);

    // Initializing yarp
    Network::init();

    // see if user has supplied audio device
    Property p;
    if (argc>1) {
        p.fromCommand(argc,argv);
    }



    // otherwise default device is "portaudio"
    if (!p.check("device")) {
        p.put("device","portaudio");
        p.put("write",1);
        p.put("delay",1);
    }

    // start the echo service running
    Echo echo;
    if (p.check("filename")) {
        echo.fname = p.find("filename").asString();
    }
    if (p.check("muted")) {
        echo.mute();
    }
    echo.open(p);

    while (!interrupt) {
        sleep(1);
    }

    echo.close();
    Network::fini();

    return 0;
}

