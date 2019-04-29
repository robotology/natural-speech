#include <iostream>
#include <unistd.h>

#include <yarp/os/RateThread.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/RFModule.h>
#include <yarp/sig/SoundFile.h>

using namespace std;

using namespace yarp::os;
using namespace yarp::sig::file;

class SoundPlayerThread: public yarp::os::RateThread {
    public:
        SoundPlayerThread(int period, yarp::os::Port* sound_port):
            RateThread(period), sound_port_(sound_port) {
            start();
            suspend();
        }

        bool play(string fname) {
            fname_ = fname;
            if (!read(snd_, fname.c_str()))
                return false;
            next_sample_ = 0;
            setRate(4096.0 / snd_.getFrequency() * 1000);
            resume();
            return true;
        }

        void run() {
            yarp::sig::Sound sub;
            int last_sample;

            last_sample = next_sample_ + 4096;
            sub = snd_.subSound(next_sample_, last_sample);
            sound_port_->write(sub);
            next_sample_ = next_sample_ + 4096;
            if (next_sample_ >= snd_.getSamples())
                suspend();
        }

        private:
            string fname_;
            int next_sample_;
            yarp::sig::Sound snd_;
            yarp::os::Port* sound_port_;
};

class SoundPlayerModule: public yarp::os::RFModule {
    public:
        bool configure(ResourceFinder &rf) {
            if (!cmd_port_.open("/cmd:io"))
                cerr << getName() << ": unable to open port /cmd:io" << endl;
            if (!sound_port_.open("/sound:o"))
                cerr << getName() << ": unable to open port /sound:o" << endl;
            thread_ = new SoundPlayerThread(1000, &sound_port_);
            return true;
        }

        bool updateModule() {
            Bottle cmd, reply;
            bool ok;

            cmd_port_.read(cmd);
            if (execute_cmd(cmd))
                reply.addString("ACK");
            else
                reply.addString("NACK");
            cmd_port_.reply(reply);
            return true;
        }

        bool execute_cmd(Bottle cmd) {
            string fname;
            yarp::sig::Sound s;

            if (cmd.size() != 2 || cmd.get(0).asString() != "play")
                return false;
            fname = cmd.get(1).asString();
            return thread_->play(fname);
        }

        bool interruptModule() {
            cmd_port_.interrupt();
            sound_port_.interrupt();
            return true;
        }

        bool close() {
            cmd_port_.close();
            sound_port_.close();
            return true;
        }

    private:
        yarp::os::RpcServer cmd_port_;
        yarp::os::Port sound_port_;
        SoundPlayerThread* thread_;
};

int main(int argc, char *argv[]) {
    ResourceFinder rf;
    rf.configure(argc, argv);
    SoundPlayerModule spm;
    return spm.runModule(rf);
}
