#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using fst::ArcIterator;

void MakePhoneWindow(const std::vector<int32> transcript, const std::vector< std::vector<int32> > lexicon_to_phone_map, bool insert_sil, int N, int P, std::vector< std::vector<int32> > *phone_windows) {
    std::vector<int32> phone_sequece;
    for (int i = 0; i < P; i++) {
        phone_sequece.push_back(0);
    }
    for (int i = 0; i < transcript.size(); i++) {
        if (insert_sil) {
            for (int j = 0; j < lexicon_to_phone_map[1].size(); j++) {
                phone_sequece.push_back(lexicon_to_phone_map[1][j]);
            }
        }
        for (int j = 0; j < lexicon_to_phone_map[i].size(); j++) {
            phone_sequece.push_back(lexicon_to_phone_map[transcript[i]][j]);
        }
    }
    if (insert_sil) {
        for (int j = 0; j < lexicon_to_phone_map[1].size(); j++) {
            phone_sequece.push_back(lexicon_to_phone_map[1][j]);
        }
    }
    for (int i = 0; i < N-P-1; i++) {
        phone_sequece.push_back(0);
    }
    for (int i = P; i < phone_sequece.size()-(N-P-1); i++) {
        std::vector<int32> p;
        for (int j = -P; j < N-P; j++) {
            p.push_back(phone_sequece[i+j]);
        }
        phone_windows->push_back(p);
    }
}

VectorFst<StdArc>* Connect(const std::vector<const VectorFst<StdArc>* > &fsts ) {
    typedef typename StdArc::Weight Weight;
    typedef typename StdArc::StateId StateId;
    typedef typename StdArc::Label Label;
    VectorFst<StdArc> *ans = new VectorFst<StdArc>;
    StateId start_state = ans->AddState();  // = 0.
    ans->SetStart(start_state);
  
    StateId pre_state = start_state; 
    for (Label i = 0; i < static_cast<Label>(fsts.size()); i++) {
      const VectorFst<StdArc> *fst = fsts[i];
      if (fst == NULL) continue;
  
      StateId fst_num_states = fst->NumStates();
      StateId fst_start_state = fst->Start();
  
      if (fst_start_state == fst::kNoStateId)
          continue;  // empty fst.
      std::vector<StateId> state_map(fst_num_states);  // fst state -> ans state
      for (StateId s = 0; s < fst_num_states; s++) {
          state_map[s] = ans->AddState();
      }
      StdArc newarc(0, 0, Weight::One(), state_map[0]);
      ans->AddArc(pre_state, newarc);
      
      for (StateId s = 0; s < fst_num_states; s++) {
        // Add arcs out of state s.

        for (ArcIterator<VectorFst<StdArc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
          const StdArc &arc = aiter.Value();
          Label olabel = arc.ilabel;
          StdArc newarc(arc.ilabel, olabel, arc.weight, state_map[arc.nextstate]);
          ans->AddArc(state_map[s], newarc);
        }
        pre_state = state_map[s];
      }
    }
    ans->SetFinal(pre_state, Weight::One());
    return ans;
}

int main(int argc, char *argv[]) {
  try {

    const char *usage =
        "Creates training graphs (without transition-probabilities, by default)\n"
        "\n"
        "Usage:   compile-train-graphs [options] <tree-in> <model-in> "
        "<lexicon-fst-in> <transcriptions-rspecifier> <graphs-wspecifier>\n"
        "e.g.: \n"
        " compile-train-graphs tree 1.mdl lex.int "
        "'ark:sym2int.pl -f 2- words.txt text|' ark:graphs.fsts\n";
    ParseOptions po(usage);
    
    HTransducerConfig hcfg;
    int32 batch_size = 250;
    std::string disambig_rxfilename;
    hcfg.Register(&po);

    po.Register("batch-size", &batch_size,
                "Number of FSTs to compile at a time (more -> faster but uses "
                "more memory.  E.g. 500");
    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1);
    std::string model_rxfilename = po.GetArg(2);
    std::string lex_rxfilename = po.GetArg(3);
    std::string transcript_rspecifier = po.GetArg(4);
    std::string fsts_wspecifier = po.GetArg(5);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    // need VectorFst because we will change it by adding subseq symbol.
    std::vector< std::vector<int32> > lexicon_to_phone_map;
    std::ifstream in(lex_rxfilename, std::ios::in);

    // read lexicon file 
    std::string line;
    int num;
    int linenum = 0;
    while (getline(in, line)) {
        std::istringstream iss(line);
        if (iss >> num ) {
            lexicon_to_phone_map.push_back(std::vector<int32>());
            if (!(num==linenum)) {
                KALDI_ERR << "the line number is not same as the counter, " 
                          << num << " VS. " << linenum << std::endl;
            }
        }
        while (iss >> num) {
            lexicon_to_phone_map[linenum].push_back(num);
        }
        ++linenum;
    }

    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;

    SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);
    HmmCacheType cache;
    // "cache" is an optimization that prevents GetHmmAsFst repeating work

    int num_succeed = 0, num_fail = 0;
    // to test more parts of the code.
    for (; !transcript_reader.Done(); transcript_reader.Next()) {
        std::string key = transcript_reader.Key();
        const std::vector<int32> &transcript = transcript_reader.Value();
        // here we made the phone_window for H
        std::vector< std::vector<int32> > phone_windows;
        MakePhoneWindow(transcript, lexicon_to_phone_map, false, 1, 0, &phone_windows);
        std::vector<const VectorFst<StdArc>* > fsts(phone_windows.size(), NULL);
        for (int i = 0; i < phone_windows.size(); i++) {
            VectorFst<StdArc> *fst = GetHmmAsFst(phone_windows[i],
                                        ctx_dep,
                                        trans_model,
                                        hcfg,
                                        &cache);
            fsts[i] = fst;
        
        }
        
        VectorFst<StdArc> *decode_fst = Connect(fsts);
      
        std::vector<int32> disambig; 
        AddSelfLoops(trans_model,
                     disambig,
                     0.0,
                     false,
                     decode_fst);
 
        if (decode_fst->Start() != fst::kNoStateId) {
            num_succeed++;
            fst_writer.Write(key, *decode_fst);
        } else {
            KALDI_WARN << "Empty decoding graph for utterance "
                       << key;
            num_fail++;
        }
    }
    KALDI_LOG << "compile-train-graphs: succeeded for " << num_succeed
              << " graphs, failed for " << num_fail;
    return (num_succeed != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

