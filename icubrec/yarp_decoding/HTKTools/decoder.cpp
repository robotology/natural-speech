/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*           Speech Vision and Robotics group                  */
/*           (now Machine Intelligence Laboratory)             */
/*           Cambridge University Engineering Department       */
/*           http://mi.eng.cam.ac.uk/                          */
/*                                                             */
/*           Entropic Cambridge Research Laboratory            */
/*           (now part of Microsoft)                           */
/*                                                             */
/* modified at:                                                */
/*                                                             */
/*           Istituto Italiano di Tecnologia                   */
/* ----------------------------------------------------------- */
/*           Copyright: Microsoft Corporation                  */
/*            1995-2000 Redmond, Washington USA                */
/*                      http://www.microsoft.com               */
/*                                                             */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            2001-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*       File: decoder.cpp  recognise or align file or audio   */
/* ----------------------------------------------------------- */

#include <iostream>
#include <yarp/os/RFModule.h>
#include <yarp/os/BufferedPort.h>

#include "config.h"
#ifdef IMKL
#include "mkl.h"
#endif
#ifdef CUDA
#include "HCUDA.h"
#endif
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HAudio.h"
#include "HWave.h"
#include "HVQ.h"
#include "HParm.h"
#include "HLabel.h"
#include "HANNet.h"
#include "HModel.h"
#include "HUtil.h"
#include "HTrain.h"
#include "HAdapt.h"
extern "C" {
#include "HMap.h"
}
#include "HFB.h"
#include "HDict.h"
#include "HNet.h"
#include "HArc.h"
#include "HFBLat.h"
#include "HRec.h"
#include "HNCache.h"

char *hvite_version = "!HVER!HVite:   3.5.0 [CUED 12/10/15]";
char *hvite_vc_id = "$Id: HVite.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $";

/* -------------------------- Trace Flags & Vars ------------------------ */

#define T_TOP 00001      /* Basic progress reporting */
#define T_OBS 00002      /* list observations */
#define T_FRS 00004      /* Frame by frame best token */
#define T_MEM 00010      /* Memory usage, start and finish */
#define T_MMU 00020      /* Memory usage after each utterance */

static int trace = 0;

/* -------------------------- Global Variables etc ---------------------- */

/* Doing what */
static int nToks = 0;             /* Number of tokens for N best */
static int nTrans = 1;            /* Number of transcriptions for N best */
static Boolean states = FALSE;    /* Keep track of state alignment */
static Boolean models = FALSE;    /* Keep track of model alignment */

/* With what */
static char *datFN;               /* Speech file */
static char *dictFn;              /* Dictionary */
static char *wdNetFn = NULL;      /* Word level lattice */
static char *hmmListFn;           /* HMMs */
static char * hmmDir = NULL;      /* directory to look for hmm def files */
static char * hmmExt = NULL;      /* hmm def file extension */
static Boolean loadLabels = FALSE; /* Load network for each file */
static Boolean loadNetworks = FALSE; /* Load network for each file */
static LabId bndId = NULL;        /* Boundary word for alignment */

/* Results and formats */
static char * labDir = NULL;      /* output label file directory */
static char * labExt = "rec";     /* output label file extension */
static char * labForm = NULL;     /* output label reformat */
static char * latForm = NULL;     /* output lattice format */
static char * labInDir = NULL;    /* input network/label file directory */
static char * labInExt = "lab";   /* input network/label file extension */
static char * latExt = NULL;      /* output lattice file extension */
static char * labFileMask = NULL; /* mask for reading lablels (lattices) */
static char * labOFileMask = NULL; /* mask for reading lablels (lattices) */
static char * latFileMask = NULL; /* mask for reading lablels (lattices) */
static char * latOFileMask = NULL; /* mask for reading lablels (lattices) */
static FileFormat dfmt=UNDEFF;    /* Data input file format */
static FileFormat ifmt=UNDEFF;    /* Label input file format */
static FileFormat ofmt=UNDEFF;    /* Label output file format */
static Boolean saveAudioOut=FALSE;/* Save rec output from direct audio */
static char * roPrefix=NULL;      /* Prefix for direct audio output name */
static char * roSuffix=NULL;      /* Suffix for direct audio output name */
static int roCounter = 0;         /* Counter for audio output name */
static Boolean replay = FALSE;    /* enable audio replay */

/* Language model */
static double lmScale = 1.0;      /* bigram and log(1/NSucc) scale factor */
static LogDouble wordPen = 0.0;   /* inter model propagation log prob */
static double prScale = 1.0;      /* pronunciation scale factor */

/* Pruning */
static LogDouble genBeam = -LZERO;/* genBeam threshold */
static LogDouble genBeamInc  = 0.0;       /* increment         */
static LogDouble genBeamLim = -LZERO;     /* max value       */
static LogDouble nBeam = 0.0;     /* nBeam threshold */
static LogDouble wordBeam = -LZERO;/* word-end pruning threshold */
static LogFloat tmBeam = 10.0;    /* tied mix prune threshold */
static int maxActive = 0;         /* max active phone instances */

/* Global variables */
static Observation obs;           /* current observation */
static HMMSet hset;               /* the HMM set */
static Vocab vocab;               /* the dictionary */
static Lattice *wdNet;            /* the word level recognition network */
static PSetInfo *psi;             /* Private data used by HRec */
static VRecInfo *vri;             /* Visible HRec Info */
static int maxM = 0;              /* max mixtures in any model */
static int maxMixInS[SMAX];       /* array[1..swidth[0]] of max mixes */

/* Global adaptation variables */
static int update = 0;            /* Perfom MLLR & update every n utts */
static UttInfo *utt;              /* utterance info for state/frame align */
static FBInfo *fbInfo;            /* forward-backward info for alignment */
static PSetInfo *alignpsi;        /* Private data used by HRec */
static VRecInfo *alignvri;        /* Visible HRec Info */
static Boolean saveBinary=FALSE;  /* Save tmf in binary format */

/* cz277 - ANN */
/*static int batchSamples;*/
static LabelInfo labelInfo;
static DataCache *cache[SMAX];

/* Heaps */
static MemHeap ansHeap;
static MemHeap modelHeap;
static MemHeap netHeap;
static MemHeap bufHeap;
static MemHeap repHeap;
static MemHeap regHeap;
/* cz277 - ANN */
static MemHeap cacheHeap;

/* information about transforms */
static XFInfo xfInfo;

/* ---------------- Configuration Parameters --------------------- */

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;            /* total num params */

using namespace std;
using namespace yarp::os;

class MyModule:public RFModule
{
public:
    int argc;
    char **argv;

private:
    RpcServer handlerPort; //a port to handle messages
    BufferedPort<Bottle> speechOutPort;
    bool hvite_initialized;

public:

    double getPeriod()
    {
        return 1; //module periodicity (seconds)
    }

    /*
    * This is our main function. Will be called periodically every getPeriod() seconds.
    */
    bool updateModule()
    {
        return true;
    }

    bool respond(const Bottle& command, Bottle& reply)
    {
        cout<<"Got something, echo is on"<<endl;
        if (command.get(0).asString()=="quit")
            return false;
        else if (command.get(0).asString()=="recognize") {
            //int argc = 16;
            //char* argv[] = {"HVite", "-C", "config", "-H", "mmf", "-i", "recout.mlf", "-w", "wdnet_upper", "-p", "4.0", "-s", "15", "dict_sp", "tiedlist", "speech.wav"};
            initHVite(argc, argv);
            DoRecognition();
            cleanHVite();
            reply.addString("OK");
        }
        else
            reply=command;
        return true;
    }

    /*
    * Configure function. Receive a previously initialized
    * resource finder object. Use it to configure your module.
    * Open port and attach it to message handler.
    */
    bool configure(yarp::os::ResourceFinder &rf)
    {
        if (!handlerPort.open("/decoder/rpc:i"))
            return false;
        if (!speechOutPort.open("/decoder/speech:o"))
            return false;

        attach(handlerPort);
        return true;
    }

    /*
    * Interrupt function.
    */
    bool interruptModule()
    {
        cout<<"Interrupting your module, for port cleanup"<<endl;
        return true;
    }

    /*
    * Close function, to perform cleanup.
    */
    bool close()
    {
        cout<<"Calling close function\n";
        handlerPort.close();
        speechOutPort.close();
        /*if (hvite_initialized)
            cleanHVite();*/
        return true;
    }

    void initHVite(int argc, char *argv[]) {
        char *s;
        int i;

        if(InitShell(argc,argv,hvite_version,hvite_vc_id)<SUCCESS)
          HError(3200,"HVite: InitShell failed");

        InitMem();   InitLabel();
        InitMath();  InitSigP();
        InitWave();  InitAudio();
        InitVQ();    InitModel();
        /* cz277 - ANN */
        #ifdef CUDA
        InitCUDA();
        #endif
        InitANNet();

        if(InitParm()<SUCCESS)
          HError(3200,"HVite: InitParm failed");

        InitDict();
        InitNet();   InitRec();
        InitUtil();
        /* cz277 - xform */
        /*InitAdapt(&xfInfo);*/
        InitAdapt();
        InitXFInfo(&xfInfo);

        InitMap();
        InitNCache();

        if (!InfoPrinted() && NumArgs() == 0)
          ReportUsage();
        if (NumArgs() == 0) Exit(0);

        SetConfParms();
        CreateHeap(&modelHeap, "Model heap",  MSTAK, 1, 0.0, 100000, 800000 );
        CreateHMMSet(&hset,&modelHeap,TRUE);

        while (NextArg() == SWITCHARG) {
          s = GetSwtArg();
          if (strlen(s)!=1)
             HError(3219,"HVite: Bad switch %s; must be single letter",s);
          switch(s[0]){
          case 'a':
             loadLabels=TRUE; break;
          case 'b':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Utterance boundary word expected");
             bndId = GetLabId(GetStrArg(),TRUE); break;
          case 'c':
             tmBeam = GetChkedFlt(0.0,1000.0,s); break;
          case 'd':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: HMM definition directory expected");
             hmmDir = GetStrArg(); break;
          case 'e':
             saveAudioOut=TRUE; break;
          case 'f':
             states=TRUE; break;
          case 'g':
             replay=TRUE; break;
          case 'i':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Output MLF file name expected");
             /* if(SaveToMasterfile(GetStrArg())<SUCCESS)
                HError(3214,"HCopy: Cannot write to MLF"); */
             SaveToMasterfile(GetStrArg());
             break;
          case 'k':
         xfInfo.useInXForm = TRUE;
         break;
          case 'j':
             if (NextArg()!=INTARG)
                HError(3219,"HVite: No. of files per online adaptation step expected");
             update = GetChkedInt(1,256,s);
             break;
          case 'l':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Label file directory expected");
             labDir = GetStrArg(); break;
          case 'm':
             models=TRUE; break;
          case 'n':
             nToks = GetChkedInt(2,MAX_TOKS,s);
             if (NextArg()==FLOATARG || NextArg()==INTARG)
                nTrans = GetChkedInt(1,10000,s);
             else
                nTrans = 1;
             break;
          case 'o':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Output label format expected");
             labForm = GetStrArg(); break;
          case 'p':
             wordPen = GetChkedFlt(-1000.0,1000.0,s);  break;
          case 'q':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Output lattice format expected");
             latForm = GetStrArg(); break;
          case 'r':
             prScale = GetChkedFlt(0.0,1000.0,s);  break;
          case 's':
             lmScale = GetChkedFlt(0.0,1000.0,s);  break;
          case 't':
             genBeam = GetChkedFlt(0,1.0E20,s);
         if (genBeam == 0.0)
            genBeam = -LZERO;
             if (NextArg()==FLOATARG || NextArg()==INTARG) {
                 genBeamInc = GetChkedFlt(0.0,1.0E20,s);
                 genBeamLim = GetChkedFlt(0.0,1.0E20,s);
                 if (genBeamLim < (genBeam + genBeamInc)) {
                    genBeamLim = genBeam; genBeamInc = 0.0;
                 }
              }
              else {
                 genBeamInc = 0.0;
                 genBeamLim = genBeam;
              }
              break;
          case 'w':
             if (NextArg()!=STRINGARG)
                loadNetworks=TRUE;
             else {
                wdNetFn = GetStrArg();
                if (strlen(wdNetFn)==0) {
                   wdNetFn=NULL;
                   loadNetworks=TRUE;
                }
             }
             break;
          case 'u':
             maxActive = GetChkedInt(0,100000,s); break;
          case 'v':
             wordBeam = GetChkedFlt(0,1.0E20,s);
             if (wordBeam == 0.0)
                wordBeam = -LZERO;
             break;
          case 'x':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: HMM file extension expected");
             hmmExt = GetStrArg(); break;
          case 'y':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Output label file extension expected");
             labExt = GetStrArg(); break;
          case 'z':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Lattice output file extension expected");
             latExt = GetStrArg(); break;
          case 'F':
             if (NextArg() != STRINGARG)
                HError(3219,"HVite: Data File format expected");
             if((dfmt = Str2Format(GetStrArg())) == ALIEN)
                HError(-3289,"HVite: Warning ALIEN Input file format set");
             break;
          case 'G':
             if (NextArg() != STRINGARG)
                HError(3219,"HVite: Source Label File format expected");
             if((ifmt = Str2Format(GetStrArg())) == ALIEN)
                HError(-3289,"HVite: Warning ALIEN Input file format set");
             break;
          case 'H':
             if (NextArg() != STRINGARG)
                HError(3219,"HVite: MMF File name expected");
             AddMMF(&hset,GetStrArg());
             break;
          case 'I':
             if (NextArg() != STRINGARG)
                HError(3219,"HVite: MLF file name expected");
             LoadMasterFile(GetStrArg()); break;
          case 'L':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Label/network file directory expected");
             labInDir = GetStrArg(); break;
          case 'P':
             if (NextArg() != STRINGARG)
                HError(3219,"HVite: Target Label File format expected");
             if((ofmt = Str2Format(GetStrArg())) == ALIEN)
                HError(-3289,"HVite: Warning ALIEN Label output file format set");
             break;
          case 'B':
             saveBinary = TRUE;
             break;
          case 'T':
             trace = GetChkedInt(0,511,s); break;
          case 'X':
             if (NextArg()!=STRINGARG)
                HError(3219,"HVite: Input label/network file extension expected");
             labInExt = GetStrArg(); break;
          case 'h':
        if (NextArg()!=STRINGARG)
          HError(1,"Speaker name pattern expected");
        xfInfo.outSpkrPat = GetStrArg();
        if (NextArg()==STRINGARG) {
          xfInfo.inSpkrPat = GetStrArg();
          if (NextArg()==STRINGARG)
            xfInfo.paSpkrPat = GetStrArg();
        }
        if (NextArg() != SWITCHARG)
          HError(2319,"HERest: cannot have -h as the last option");
        break;
          case 'E':
             if (NextArg()!=STRINGARG)
                HError(2319,"HERest: parent transform directory expected");
         xfInfo.usePaXForm = TRUE;
             xfInfo.paXFormDir = GetStrArg();
             if (NextArg()==STRINGARG)
           xfInfo.paXFormExt = GetStrArg();
         if (NextArg() != SWITCHARG)
           HError(2319,"HVite: cannot have -E as the last option");
             break;
          case 'J':
             if (NextArg()!=STRINGARG)
                HError(2319,"HERest: input transform directory expected");
             AddInXFormDir(&hset,GetStrArg());
             if (NextArg()==STRINGARG)
           xfInfo.inXFormExt = GetStrArg();
         if (NextArg() != SWITCHARG)
           HError(2319,"HVite: cannot have -J as the last option");
             break;
          case 'K':
             if (NextArg()!=STRINGARG)
                HError(2319,"HVite: output transform directory expected");
             xfInfo.outXFormDir = GetStrArg();
         xfInfo.useOutXForm = TRUE;
             if (NextArg()==STRINGARG)
           xfInfo.outXFormExt = GetStrArg();
         if (NextArg() != SWITCHARG)
           HError(2319,"HVite: cannot have -K as the last option");
             break;
          default:
             HError(3219,"HVite: Unknown switch %s",s);
          }
        }

        if (NextArg()!=STRINGARG)
          HError(3219,"HVite: Dictionary file name expected");
        dictFn = GetStrArg();
        if (NextArg()!=STRINGARG)
          HError(3219,"HVite: HMM list  file name expected");
        hmmListFn = GetStrArg();

        #ifndef PHNALG
        if ((states || models) && nToks>1)
          HError(3230,"HVite: Alignment using multiple tokens is not supported");
        #endif
        if (NumArgs()==0 && wdNetFn==NULL)
          HError(3230,"HVite: Network must be specified for recognition from audio");
        if (loadNetworks && loadLabels)
          HError(3230,"HVite: Must choose either alignment from network or labels");
        if (nToks>1 && latExt==NULL && nTrans==1)
          HError(-3230,"HVite: Performing nbest recognition with no nbest output");
        if (nToks > 1 && latExt != NULL && nTrans > 1)
          HError(-3230,"HVite: Performing nbest recognition with 1-best and latttices output");
        if ((update>0) && (!xfInfo.useOutXForm))
          HError(3230,"HVite: Must use -K option with incremental adaptation");


        #ifdef CUDA
        StartCUDA();
        #endif
        /* cz277 - 151020 */
        #ifdef MKL
        StartMKL();
        #endif
        Initialise();
        #ifdef CUDA
        ShowGPUMemUsage();
        #endif
        hvite_initialized = true;
    }

    void cleanHVite() {
        /* Free up and we are done */

        if (trace & T_MEM) {
          printf("Memory State on Completion\n");
          PrintAllHeapStats();
        }

        DeleteVRecInfo(vri);
        ResetHeap(&netHeap);
        FreePSetInfo(psi);
        UpdateSpkrStats(&hset,&xfInfo, NULL);
        ResetHeap(&regHeap);
        ResetHeap(&modelHeap);

        /* cz277 - ANN */
        /* remove the ANNSet matrices and vectors */
        if (hset.annSet != NULL) {
           FreeANNSet(&hset);
           for (int i = 1; i <= hset.swidth[0]; ++i) {
              FreeCache(cache[i]);
           }
        }
        #ifdef CUDA
        StopCUDA();
        #endif
    }


    /* Initialise: set up global data structures */
    void Initialise(void)
    {
       Boolean eSep;
       int s;
       /* cz277 - ANN */
       FILE *script;
       int scriptcount;

       /* Load hmms, convert to inverse DiagC */
       if(MakeHMMSet(&hset,hmmListFn)<SUCCESS)
          HError(3228,"Initialise: MakeHMMSet failed");
       if(LoadHMMSet(&hset,hmmDir,hmmExt)<SUCCESS)
          HError(3228,"Initialise: LoadHMMSet failed");
       ConvDiagC(&hset,TRUE);

       /* Create observation and storage for input buffer */
       SetStreamWidths(hset.pkind,hset.vecSize,hset.swidth,&eSep);
       obs=MakeObservation(&gstack,hset.swidth,hset.pkind,
                           (Boolean)(hset.hsKind==DISCRETEHS),eSep);	/* TODO: for Tandem system, might need an extra obs */

       /* sort out masks just in case using adaptation */
       if (xfInfo.inSpkrPat == NULL) xfInfo.inSpkrPat = xfInfo.outSpkrPat;
       if (xfInfo.paSpkrPat == NULL) xfInfo.paSpkrPat = xfInfo.outSpkrPat;

       if (xfInfo.useOutXForm || (update>0)) {
          CreateHeap(&regHeap,   "regClassStore",  MSTAK, 1, 0.5, 1000, 8000 );
          /* This initialises things - temporary hack - THINK!! */
          CreateAdaptXForm(&hset, "tmp");
          /* initialise structures for the f-b frame-state alignment pass */
          utt = (UttInfo *) New(&regHeap, sizeof(UttInfo));
          fbInfo = (FBInfo *) New(&regHeap, sizeof(FBInfo));
          /* initialise a recogniser for frame/state alignment purposes */
          alignpsi=InitPSetInfo(&hset);
          alignvri=InitVRecInfo(alignpsi,1,TRUE,FALSE);
          SetPruningLevels(alignvri,0,genBeam,-LZERO,0.0,tmBeam);
          InitUttInfo(utt, FALSE);
          InitialiseForBack(fbInfo, &regHeap, &hset,
                            (UPDSet) (UPXFORM), genBeam*2.0, genBeam*2.0,
                            genBeam*4.0+1.0, 10.0);
          utt->twoDataFiles = FALSE;
          utt->S = hset.swidth[0];
          AttachPreComps(&hset,hset.hmem);
       }

       CreateHeap(&bufHeap,"Input Buffer heap",MSTAK,1,0.0,50000,50000);
       CreateHeap(&repHeap,"Replay Buffer heap",MSTAK,1,0.0,50000,50000);

       maxM = MaxMixInSet(&hset);
       for (s=1; s<=hset.swidth[0]; s++)
          maxMixInS[s] = MaxMixInSetS(&hset, s);
       if (trace&T_TOP) {
          printf("Read %d physical / %d logical HMMs\n",
                 hset.numPhyHMM,hset.numLogHMM);
          /* cz277 - ANN */
          if (hset.annSet != NULL) {
             if (hset.hsKind == HYBRIDHS)
                printf("Hybrid ANN set: ");
             else
                printf("Tandem ANN set: ");
             ShowANNSet(&hset);
          }
          fflush (stdout);
       }

       SetupNMatRPLInfo(&hset);
       SetupNVecRPLInfo(&hset);

       /* Initialise recogniser */
       if (nToks>1) nBeam=genBeam;
       psi=InitPSetInfo(&hset);
       vri=InitVRecInfo(psi,nToks,models,states);

       /* Read dictionary and create storage for lattice */
       InitVocab(&vocab);
       if(ReadDict(dictFn,&vocab)<SUCCESS)
          HError(3213, "Main: ReadDict failed");
       CreateHeap(&ansHeap,"Lattice heap",MSTAK,1,0.0,4000,4000);
       if (trace & T_MEM){
          printf("Memory State After Initialisation\n");
          PrintAllHeapStats();
       }

       /* cz277 - ANN */
       /* ANN and data cache related code */
       /* set label info */
       if (hset.annSet != NULL) {
          labelInfo.labelKind = LABLK;
          labelInfo.labFileMask = NULL;
          labelInfo.labDir = labDir;
          labelInfo.labExt = labExt;
          labelInfo.latFileMask = NULL;
          labelInfo.latMaskNum = NULL;
          labelInfo.numLatDir = NULL;
          labelInfo.nNumLats = 0;
          labelInfo.numLatSubDirPat = NULL;
          labelInfo.latMaskDen = NULL;
          labelInfo.denLatDir = NULL;
          labelInfo.nDenLats = 0;
          labelInfo.denLatSubDirPat = NULL;
          labelInfo.latExt = NULL;
          /* get script info */
          script = GetTrainScript(&scriptcount);
          /* initialise the cache heap */
          CreateHeap(&cacheHeap, "cache heap", CHEAP, 1, 0, 100000000, ULONG_MAX);
          /* initialise DataCache structure */
          for (s = 1; s <= hset.swidth[0]; ++s) {
             /*cache[s] = CreateCache(&cacheHeap, script, scriptcount, (Ptr) &hset, &obs, 1, -1, NONEVK, &xfInfo, NULL, TRUE);*/
             cache[s] = CreateCache(&cacheHeap, script, scriptcount, (HMMSet*) (Ptr) &hset, &obs, 1, GetDefaultNCacheSamples(), NONEVK, &xfInfo, NULL, TRUE);
             InitCache(cache[s]);
          }
       }

    }

    void ReportUsage(void)
    {
       printf("\nUSAGE: HVite [options] VocabFile HMMList DataFiles...\n\n");
       printf(" Option                                       Default\n\n");
       printf(" -a      align from label files               off\n");
       printf(" -b s    def s as utterance boundary word     none\n");
       printf(" -c f    tied mixture pruning threshold       10.0\n");
       printf(" -d s    dir to find hmm definitions          current\n");
       printf(" -e      save direct audio rec output         off\n");
       printf(" -f      output full state alignment          off\n");
       printf(" -g      enable audio replay                  off\n");
       printf(" -h s    set speaker name pattern             *.mfc\n");
       printf(" -i s    Output transcriptions to MLF s       off\n");
       printf(" -j i    Online MLLR adaptation               off\n");
       printf("         Perform update every i utterances      \n");
       printf(" -k      use an input transform               off\n");
       printf(" -l s    dir to store label/lattice files     current\n");
       printf(" -m      output model alignment               off\n");
       printf(" -n i [N] N-best recognition (using i tokens) off\n");
       printf(" -o s    output label formating NCSTWMX       none\n");
       printf(" -p f    inter model trans penalty (log)      0.0\n");
       printf(" -q s    output lattice formating ABtvaldmn   tvaldmn\n");
       printf(" -r f    pronunciation prob scale factor      1.0\n");
       printf(" -s f    grammar scale factor                 1.0\n");
       printf(" -t f [f f] set pruning threshold             0.0\n");
       printf(" -u i    set pruning max active               0\n");
       printf(" -v f    set word end pruning threshold       0.0\n");
       printf(" -w [s]  recognise from network               off\n");
       printf(" -x s    extension for hmm files              none\n");
       printf(" -y s    output label file extension          rec\n");
       printf(" -z s    generate lattices with extension s   off\n");
       PrintStdOpts("BEFGHIJKLPSX");
       printf("\n\n");
    }

    void SetConfParms(void)
    {
       int i;
       Boolean b;
       char buf[MAXSTRLEN];

       nParm = GetConfig("HVITE", TRUE, cParm, MAXGLOBS);
       if (nParm>0){
          if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
          if (GetConfStr(cParm,nParm,"RECOUTPREFIX",buf))
             roPrefix=CopyString(&gstack,buf);
          if (GetConfStr(cParm,nParm,"RECOUTSUFFIX",buf))
             roSuffix=CopyString(&gstack,buf);
          if (GetConfBool(cParm,nParm,"SAVEBINARY",&b))
             saveBinary = b;
          if (GetConfStr(cParm,nParm,"LABFILEMASK",buf)) {
             labFileMask = CopyString(&gstack, buf);
          }
          if (GetConfStr(cParm,nParm,"LABOFILEMASK",buf)) {
             labOFileMask = CopyString(&gstack, buf);
          }
          if (GetConfStr(cParm,nParm,"LATFILEMASK",buf)) {
             latFileMask = CopyString(&gstack, buf);
          }
          if (GetConfStr(cParm,nParm,"LATOFILEMASK",buf)) {
             latOFileMask = CopyString(&gstack, buf);
          }
       }
    }

    /* DoRecognition:  use single network to recognise each input utterance */
    void DoRecognition(void)
    {
        FILE *nf;
        Network *net;
        Boolean isPipe;
        int n=0;
        AdaptXForm *incXForm;
        /* cz277 - ANN */
        char fnbuf[1024];

        if ( (nf = FOpen(wdNetFn,NetFilter,&isPipe)) == NULL)
          HError(3210,"DoRecognition: Cannot open Word Net file %s",wdNetFn);
        if((wdNet = ReadLattice(nf,&ansHeap,&vocab,TRUE,FALSE))==NULL)
          HError(3210,"DoAlignment: ReadLattice failed");
        FClose(nf,isPipe);

        if (trace&T_TOP) {
          printf("Read lattice with %d nodes / %d arcs\n",wdNet->nn,wdNet->na);
          fflush(stdout);
        }
        CreateHeap(&netHeap,"Net heap",MSTAK,1,0,
                  wdNet->na*sizeof(NetLink),wdNet->na*sizeof(NetLink));

        net = ExpandWordNet(&netHeap,wdNet,&vocab,&hset);
        ResetHeap(&ansHeap);
        if (trace&T_TOP) {
          printf("Created network with %d nodes / %d links\n",
                 net->numNode,net->numLink);  fflush(stdout);
        }
        if (trace & T_MEM){
          printf("Memory State Before Recognition\n");
          PrintAllHeapStats();
        }

        if (NumArgs()==0) {      /* Process audio */
          while(TRUE){
             printf("\nREADY[%d]>\n",++n); fflush(stdout);
         /* no input transform possible for audio input .... */
             ProcessFile(NULL,net,n,genBeam, FALSE);
             if (update > 0 && n%update == 0) {
                if (trace&T_TOP) {
                   printf("Transforming model set\n");
                   fflush(stdout);
                }
            /*
               at every stage a new transform is created - fix??
               Estimate transform and then set it up as the
               input XForm
            */
            incXForm = CreateAdaptXForm(&hset,"inc");
                TidyBaseAccs();
            GenAdaptXForm(&hset,incXForm);
                xfInfo.inXForm = GetMLLRDiagCov(incXForm);;
                SetXForm(&hset,xfInfo.inXForm);
            ApplyHMMSetXForm(&hset,xfInfo.inXForm);
             }
          }
        }
        else {                   /* Process files */
          while (NumArgs()>0) {
             if (NextArg()!=STRINGARG)
                HError(3219,"DoRecognition: Data file name expected");
             datFN = GetStrArg();
             /* cz277 - ANN */
             strcpy(fnbuf, datFN);

             if (trace&T_TOP) {
                printf("File: %s\n",fnbuf); fflush(stdout);
             }
         /* This handles the initial input transform, parent transform setting
            and output transform creation */
             if (UpdateSpkrStats(&hset, &xfInfo, fnbuf) && (!(xfInfo.useInXForm)) && (hset.semiTied == NULL)) {
                xfInfo.inXForm = NULL;
             }
             ProcessFile(fnbuf,net,n++,genBeam,FALSE);
             if (update > 0 && n%update == 0) {
                if (trace&T_TOP) {
                   printf("Transforming model set\n");
                   fflush(stdout);
                }
            /*
               at every stage a new transform is created - fix??
               Estimate transform and then set it up as the
               input XForm
            */
            incXForm = CreateAdaptXForm(&hset,"inc");
                TidyBaseAccs();
            GenAdaptXForm(&hset,incXForm);
                xfInfo.inXForm = GetMLLRDiagCov(incXForm);;
                SetXForm(&hset,xfInfo.inXForm);
            ApplyHMMSetXForm(&hset,xfInfo.inXForm);
             }
          }
        }
    }

    /* ProcessFile: process given file. If fn=NULL then direct audio */
    Boolean ProcessFile(char *fn, Network *net, int utterNum, LogDouble currGenBeam, Boolean restartable)
    {
       FILE *file;
       ParmBuf pbuf;
       BufferInfo pbinfo;
       NetNode *d;
       Lattice *lat;
       LArc *arc,*cur;
       LNode *node;
       Transcription *trans;
       MLink m;
       LogFloat lmlk,aclk;
       int s,j,tact,nFrames;
       LatFormat form;
       char *p,lfn[255],buf1[80],buf2[80],thisFN[MAXSTRLEN];
       char labfn[MAXFNAMELEN];
       Boolean enableOutput = TRUE, isPipe;
       /* cz277 - ANN */
       int uttCnt, cUttLen, uttLen, nLoaded, i;
       LELink layerElem;
       /* cz277 - clock */
       clock_t fwdStClock, fwdClock = 0, decStClock, decClock = 0, loadStClock, loadClock = 0;
       double fwdSec = 0.0, decSec = 0.0, loadSec = 0.0;
       /* cz277 - xform */
       UttElem *uttElem;

       if (fn!=NULL)
          strcpy(thisFN,fn);
       else if (fn==NULL && saveAudioOut)
          CounterFN(roPrefix,roSuffix,++roCounter,4,thisFN);
       else
          enableOutput = FALSE;

       if((pbuf = OpenBuffer(&bufHeap,fn,50,dfmt,TRI_UNDEF,TRI_UNDEF))==NULL)
          HError(3250,"ProcessFile: Config parameters invalid");

       /* Check pbuf same as hset */
       GetBufferInfo(pbuf,&pbinfo);
       if (pbinfo.tgtPK!=hset.pkind)
          HError(3231,"ProcessFile: Incompatible sample kind %s vs %s",
                 ParmKind2Str(pbinfo.tgtPK,buf1),
                 ParmKind2Str(hset.pkind,buf2));
       if (pbinfo.a != NULL && replay)  AttachReplayBuf(pbinfo.a, (int) (3*(1.0E+07/pbinfo.srcSampRate)));

       StartRecognition(vri,net,lmScale,wordPen,prScale);
       SetPruningLevels(vri,maxActive,currGenBeam,wordBeam,nBeam,tmBeam);

       tact=0;nFrames=0;

       /* cz277 - ANN */
       if (hset.annSet == NULL) {
          StartBuffer(pbuf);
          while(BufferStatus(pbuf)!=PB_CLEARED) {
             ReadAsBuffer(pbuf,&obs);
             if (trace&T_OBS) PrintObservation(nFrames,&obs,13);

             if (hset.hsKind==DISCRETEHS){
                for (s=1; s<=hset.swidth[0]; s++){
                   if( (obs.vq[s] < 1) || (obs.vq[s] > maxMixInS[s]))
                      HError(3250,"ProcessFile: Discrete data value [ %d ] out of range in stream [ %d ] in file %s",obs.vq[s],s,fn);
                }
             }

             ProcessObservation(vri,&obs,-1,xfInfo.inXForm);

             if (trace & T_FRS) {
                for (d=vri->genMaxNode,j=0;j<30;d=d->links[0].node,j++)
                   if (d->type==n_word) break;
                if (d->type==n_word){
                   if (d->info.pron==NULL) p=":bound:";
                   else p=d->info.pron->word->wordName->name;
                }
                else p=":external:";
                m=FindMacroStruct(&hset,'h',vri->genMaxNode->info.hmm);
                printf("Optimum @%-4d HMM: %s (%s)  %d %5.3f\n",
                       vri->frame,m->id->name,p,
                       vri->nact,vri->genMaxTok.like/vri->frame);
                fflush(stdout);
             }
             nFrames++;
             tact+=vri->nact;
          }
       }
       else {
          /* get utterance name in cache */
          if (strcmp(GetCurUttName(cache[1]), fn) != 0)
             HError(3234, "Mismatched utterance in the cache and script file");
          uttElem = GetCurUttElem(cache[1]);	/* cz277 - xform */
          /* install the current replaceable parts */
          InstallOneUttNMatRPLs(uttElem);
          InstallOneUttNVecRPLs(uttElem);
          /* check the observation vector number */
          uttCnt = 1;
          uttLen = ObsInBuffer(pbuf);
          cUttLen = GetCurUttLen(cache[1]);
          if (cUttLen != uttLen)
             HError(3292, "Unequal utterance length in the cache and the original feature file");
          while (nFrames < uttLen) {
             /* load a data batch */
             loadStClock = clock();  /* cz277 - clock */
             for (s = 1; s <= hset.swidth[0]; ++s) {
                FillAllInpBatch(cache[s], &nLoaded, &uttCnt);
                /* cz277 - mtload */
                /*UpdateCacheStatus(cache[s]);*/
                LoadCacheData(cache[s]);
             }
             /*if (nLoaded != 1)
                 HError(9999, "HVite is only able to process frame by frame");*/
             loadClock += clock() - loadStClock;   /* cz277 - clock */
             /* forward these frames */
             fwdStClock = clock();   /* cz277 - clock */
             ForwardProp(hset.annSet, nLoaded, cache[1]->CMDVecPL);
             /*SetBatchIndex(GetBatchIndex() + 1);*/
             /* apply log transform */
             for (s = 1; s <= hset.swidth[0]; ++s) {
                layerElem = hset.annSet->outLayers[s];
                ApplyLogTrans(layerElem->yFeaMats[1], nLoaded, layerElem->nodeNum, hset.annSet->llhMat[s]);	/* cz277 - many */
                AddNVectorTargetPen(hset.annSet->llhMat[s], hset.annSet->penVec[s], nLoaded, hset.annSet->llhMat[s]);
    #ifdef CUDA
                SyncNMatrixDev2Host(hset.annSet->llhMat[s]);
    #endif
             }
             fwdClock += clock() - fwdStClock;   /* cz277 - clock */
             /* load the ANN outputs into dec->cacheVecs */
             decStClock = clock();   /* cz277 - clock */
             for (i = 0; i < nLoaded; ++i) {
                 LoadCacheVec(&obs, i, &hset);
                 /* decode current frame */
                 ProcessObservation(vri, &obs, -1, xfInfo.inXForm);
                 if (trace & T_FRS) {
                     for (d = vri->genMaxNode, j = 0; j < 30; d = d->links[0].node, j++)
                         if (d->type == n_word) break;
                     if (d->type == n_word) {
                         if (d->info.pron == NULL) p = ":bound:";
                         else p = d->info.pron->word->wordName->name;
                     }
                     else p = ":external:";
                     m = FindMacroStruct(&hset, 'h', vri->genMaxNode->info.hmm);
                     printf("Optimum @%-4d HMM: %s (%s)  %d %5.3f\n",
                            vri->frame, m->id->name, p, vri->nact,
                            vri->genMaxTok.like / vri->frame);
                     fflush(stdout);
                 }
                 tact += vri->nact;
                 /* increate nFrames */
                 ++nFrames;
             }
             decClock += clock() - decStClock;   /* cz277 - clock */
             /* cz277 - 1007 */
             /*SetBatchIndex(GetBatchIndex() + 1);*/
          }
          /* cz277 - mtload */
          for (s = 1; s <= hset.swidth[0]; ++s) {
             UnloadCacheData(cache[s]);
          }
          /* reset the replaceable parts */
          ResetNMatRPL();
          ResetNVecRPL();
       }

       lat=CompleteRecognition(vri,pbinfo.tgtSampRate/10000000.0,&ansHeap);

       if (lat==NULL) {
          if ((trace & T_TOP) && fn != NULL){
             if (restartable)
                printf("No tokens survived to final node of network at beam %.1f\n", currGenBeam);
             else
                printf("No tokens survived to final node of network\n");
             fflush(stdout);
          } else if (fn==NULL){
             printf("Sorry [%d frames]?\n",nFrames);fflush(stdout);
          }
          if (pbinfo.a != NULL && replay)  ReplayAudio(pbinfo);
          CloseBuffer(pbuf);
          return FALSE;
       }

       if (vri->noTokenSurvived && restartable)
          return FALSE;

       if (vri->noTokenSurvived && trace & T_TOP) {
          printf("No tokens survived to final node of network\n");
          printf("  Output most likely partial hypothesis within network\n");
          fflush(stdout);
       }

       lat->utterance=thisFN;
       lat->net=wdNetFn;
       lat->vocab=dictFn;

       if (trace & T_TOP || fn==NULL) {
          node=NULL;
          for (j=0;j<lat->nn;j++) {
             node=lat->lnodes+j;
             if (node->pred==NULL) break;
             node=NULL;
          }
          aclk=lmlk=0.0;
          while(node!=NULL) {
             for (arc=NULL,cur=node->foll;cur!=NULL;cur=cur->farc) arc=cur;
             if (arc==NULL) break;
             if (arc->end->word!=NULL)
                printf("%s ",arc->end->word->wordName->name);
             aclk+=arc->aclike+arc->prlike*lat->prscale;
             lmlk+=arc->lmlike*lat->lmscale+lat->wdpenalty;
             node=arc->end;
          }
          printf(" ==  [%d frames] %.4f [Ac=%.1f LM=%.1f] (Act=%.1f)\n",nFrames,
                 (aclk+lmlk)/nFrames, aclk,lmlk,(float)tact/nFrames);
          fflush(stdout);
       }

        /* cz277 - clock */
        if (hset.annSet != NULL) {
           fwdSec = fwdClock / (double) CLOCKS_PER_SEC;
           decSec = decClock / (double) CLOCKS_PER_SEC;
           loadSec = loadClock / (double) CLOCKS_PER_SEC;
           printf("\tForwarding time is %f\n", fwdSec);
           printf("\tDecoding time is %f\n", decSec);
           printf("\tCache loading time is %f\n", loadSec);
           fflush(stdout);
        }

       if (pbinfo.a != NULL && replay)  ReplayAudio(pbinfo);

       /* accumulate stats for online unsupervised adaptation
          only if a token survived */
       if ((lat != NULL) &&  (!vri->noTokenSurvived) && ((update > 0) || (xfInfo.useOutXForm)))
          DoOnlineAdaptation(lat, pbuf, nFrames);

       if (enableOutput){
            if (nToks>1 && latExt!=NULL) {
             if (latOFileMask) {
                if (!MaskMatch (latOFileMask, labfn, thisFN))
                   HError(2319,"HLRescore: LATOFILEMASK %s has no match with segemnt %s", latOFileMask, thisFN);
             } else
                strcpy (labfn, thisFN);
             MakeFN(labfn,labDir,latExt,lfn);
             if ((file=FOpen(lfn,NetOFilter,&isPipe))==NULL)
                HError(3211,"ProcessFile: Could not open file %s for lattice output",lfn);
             if (latForm==NULL)
                form=HLAT_DEFAULT;
             else {
                for (p=latForm,form=0;*p!=0;p++) {
                   switch (*p) {
                   case 'A': form|=HLAT_ALABS; break;
                   case 'B': form|=HLAT_LBIN; break;
                   case 't': form|=HLAT_TIMES; break;
                   case 'v': form|=HLAT_PRON; break;
                   case 'a': form|=HLAT_ACLIKE; break;
                   case 'l': form|=HLAT_LMLIKE; break;
                   case 'd': form|=HLAT_ALIGN; break;
                   case 'm': form|=HLAT_ALDUR; break;
                   case 'n': form|=HLAT_ALLIKE; break;
                   case 'r': form|=HLAT_PRLIKE; break;
                   }
                }
             }
             if(WriteLattice(lat,file,form)<SUCCESS)
                HError(3214,"ProcessFile: WriteLattice failed");

             FClose(file,isPipe);
            }

            /* only output 1-best transcription if generating lattices */
            if (nTrans > 1 && latExt != NULL)
             trans=TranscriptionFromLattice(&ansHeap,lat,1);
            /* output N-best transcriptions as usual */
            else
            trans=TranscriptionFromLattice(&ansHeap,lat,nTrans);

            if (labForm!=NULL)
             FormatTranscription(trans,pbinfo.tgtSampRate,states,models,
                                 (Boolean)(strchr(labForm,'X')!=NULL),
                                 (Boolean)(strchr(labForm,'N')!=NULL),(Boolean)(strchr(labForm,'S')!=NULL),
                                 (Boolean)(strchr(labForm,'C')!=NULL),(Boolean)(strchr(labForm,'T')!=NULL),
                                 (Boolean)(strchr(labForm,'W')!=NULL),(Boolean)(strchr(labForm,'M')!=NULL));

            if (labOFileMask) {
             if (!MaskMatch (labOFileMask, labfn, thisFN))
                   HError(2319,"HLRescore: LABOFILEMASK %s has no match with segemnt %s", labOFileMask, thisFN);
            } else
             strcpy (labfn, thisFN);
            MakeFN(labfn,labDir,labExt,lfn);
            /* if(LSave(lfn,trans,ofmt)<SUCCESS)
             HError(3214,"ProcessFile: Cannot save file %s", lfn); */
            Bottle& b = speechOutPort.prepare();
            b.clear();
            b = transcr2string(trans);
            speechOutPort.write();
            Dispose(&ansHeap,trans);
       }
       Dispose(&ansHeap,lat);
       CloseBuffer(pbuf);
       if (trace & T_MMU){
          printf("Memory State after utter %d\n",utterNum);
          PrintAllHeapStats();
       }

       return (Boolean)!vri->noTokenSurvived;
    }

    Bottle transcr2string(Transcription *t) {
        Bottle result;
        int i,j;
        LabList *ll;
        LLink p,hd;
        LabId id;
        Boolean hasScores[100];

        ll = t->head;
        for (i=1; i<=t->numLists; i++,ll=ll->next){
            hd = ll->head;
            /* See which columns have scores */
            for (j=0; j<=ll->maxAuxLab; j++)
                hasScores[j] = FALSE;
            for (p = hd->succ; p->succ != NULL; p = p->succ){
                if (p->score != 0.0)
                    hasScores[0] = TRUE;
                for (j=1; j<=ll->maxAuxLab; j++)
                    if (p->auxScore[j] != 0.0)
                        hasScores[j] = TRUE;
            }
            for (p = hd->succ; p->succ != NULL; p = p->succ) {
                Bottle& list = result.addList();
                list.addString(p->labid->name);
                if (hasScores[0])
                    list.addDouble(p->score);
            }
        }
        return result;
    }

    /* ReplayAudio:  replay the last audio input */
    void ReplayAudio(BufferInfo info)
    {
       AudioOut ao;

       if (info.a != NULL) {
          ao = OpenAudioOutput(&repHeap,&(info.srcSampRate));
          PlayReplayBuffer(ao, info.a);
          while (SamplesToPlay(ao) > 0 );
          CloseAudioOutput(ao);
       }
    }

    /* DoOnlineAdaptation: Perform unsupervised online adaptation
       using the recognition hypothesis as the transcription */
    int DoOnlineAdaptation(Lattice *lat, ParmBuf pbuf, int nFrames)
    {
       Transcription *modelTrans, *trans;
       BufferInfo pbinfo;
       Lattice *alignLat, *wordNet;
       Network *alignNet;
       int i;

       GetBufferInfo(pbuf,&pbinfo);
       trans=TranscriptionFromLattice(&netHeap,lat,1);
       wordNet=LatticeFromLabels(GetLabelList(trans,1),bndId,
                                 &vocab,&netHeap);
       alignNet=ExpandWordNet(&netHeap,wordNet,&vocab,&hset);

       StartRecognition(alignvri,alignNet,0.0,0.0,0.0);

       /* do forced alignment */
       for (i = 0; i < nFrames; i++) {
          ReadAsTable(pbuf, i, &obs);
          ProcessObservation(alignvri,&obs,-1,xfInfo.inXForm);
       }

       alignLat=CompleteRecognition(alignvri,
                                    pbinfo.tgtSampRate/10000000.0,
                                    &netHeap);

       if (alignvri->noTokenSurvived) {
          Dispose(&netHeap, trans);
          /* Return value 0 to indicate zero frames process failed */
          return 0;
       }
       modelTrans=TranscriptionFromLattice(&netHeap,alignLat,1);

       /* format the transcription so that it contains just the models */
       FormatTranscription(modelTrans,pbinfo.tgtSampRate,FALSE,TRUE,
                           FALSE,FALSE,TRUE,FALSE,TRUE,TRUE, FALSE);

       /* Now do the frame/state alignment accumulating MLLR statistics */
       /* set the various values in the utterance storage */
       utt->tr = modelTrans;
       utt->pbuf = pbuf;
       utt->Q = CountLabs(utt->tr->head);
       utt->T = nFrames;
       utt->ot = obs;

       /* do frame state alignment and accumulate statistics */
       fbInfo->inXForm = xfInfo.inXForm;
       fbInfo->al_inXForm = xfInfo.inXForm;
       fbInfo->paXForm = xfInfo.paXForm;
       if (!FBFile(fbInfo, utt, NULL))
         nFrames = 0;

       Dispose(&netHeap, trans);

       if (trace&T_TOP) {
          printf("Accumulated statistics...\n");
          fflush(stdout);
       }
       return nFrames;
    }

    void LoadCacheVec(Observation *obs, int shift, HMMSet *hset) {
        int s, S, i, offset;
        NMatrix *srcMat;
        FELink feaElem;
        LELink layerElem;

        if (hset->annSet == NULL)
            HError(3290, "LoadCacheVec: DataCache is only applicable for ANN related systems");

        S = hset->swidth[0];
        for (s = 1; s <= S; ++s) {
            if (hset->hsKind == HYBRIDHS) {  /* hybrid models, cache the outputs */
                layerElem = hset->annSet->outLayers[s];
                srcMat = hset->annSet->llhMat[s];
                CopyNFloatSeg2FloatSeg(srcMat->matElems + shift * layerElem->nodeNum, layerElem->nodeNum, &obs->fv[s][1]);
            }
            else if (hset->feaMix[1] != NULL) {    /* tandem models, cache the features */
                offset = 0;
                for (i = 0; i < hset->feaMix[s]->elemNum; ++i) {
                    feaElem = hset->feaMix[s]->feaList[i];
                    srcMat = feaElem->feaMats[1];	/* cz277 - many */
                    CopyNFloatSeg2FloatSeg(srcMat->matElems + feaElem->dimOff + shift * feaElem->extDim, feaElem->extDim, &obs->fv[s][1] + offset);
                    offset += feaElem->extDim;
                }
            }
            else {
                HError(3290, "LoadCacheVec: DataCache is only applicable for hybrid and tandem systems");
            }
        }

    }
};

int main(int argc, char * argv[])
{
    //Network yarp;

    MyModule module;
    module.argc = argc;
    module.argv = argv;
    ResourceFinder rf;
    rf.configure(argc, argv);
    // rf.setVerbose(true);

    cout<<"Configure module & start module..."<<endl;
    if (!module.runModule(rf))
    {
        cerr<<"Error module did not start"<<endl;
        return 1;
    }

    cout<<"Main returning..."<<endl;
    return 0;
}
