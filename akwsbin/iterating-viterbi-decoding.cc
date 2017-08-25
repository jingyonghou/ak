// gmmbin/iterating-viterbi-decoding.cc

// Copyright 2017 Microsoft Corporation (author: jyhou@nwpu-aslp.org)

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "akws/akws.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::akws;
        typedef kaldi::int32 int32;
        using fst::VectorFst;
        using fst::StdArc;
        
        const char *usage = 
            "do hmm based kws without filler model and using" 
            "iterating-biterbi-dicoding method to do kws\n"
            "Usage: iterating-viterbi-decoding <model-in> "
            "<graphs-rspecifier> <feature-rspecifier> <result_dir> \n"
            "e.g.: \n"
            "iterating-viterbi-decoding 1.mdl graphs.fsts ark:test.scp "
                                       "resultdir \n";

        ParseOptions po(usage);
        BaseFloat acoustic_scale = 1.0;
        BaseFloat transition_scale = 1.0;
        BaseFloat self_loop_scale = 1.0;

        po.Register("acoustic-scale", &acoustic_scale, 
                    "Scaling factor for acoustic likelihoods");
        po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
        po.Register("self-loop-scale", &self_loop_scale,
                    "Scale of self-loop versus non-self-loop log probs"
                    " [relative to acoustics]");

        po.Read(argc, argv);

        if (po.NumArgs() < 4) {
          po.PrintUsage();
          exit(1);
        }

        std::string model_in_filename = po.GetArg(1),
            fst_rspecifier = po.GetArg(2),
            feature_rspecifier = po.GetArg(3),
            result_dir = po.GetOptArg(4);

        TransitionModel trans_model;
        AmDiagGmm am_gmm;
        {
            bool binary;
            Input ki(model_in_filename, &binary);
            trans_model.Read(ki.Stream(), binary);
            am_gmm.Read(ki.Stream(), binary);
        }
        SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        int num_fail = 0, num_done = 0;        
 
        std::vector< VectorFst<StdArc> > fsts;
        std::vector< std::string > keyword_ids;
        for (; !fst_reader.Done(); fst_reader.Next()) {
            VectorFst<StdArc> decode_fst(fst_reader.Value());
            keyword_ids.push_back(fst_reader.Key());

            {   // Add trainsition-trobs to the FST.
                std::vector<int32> disambig_syms;  // empty.
                AddTransitionProbs(trans_model, disambig_syms,
                                   transition_scale, self_loop_scale,
                                   &decode_fst);
            }

            fsts.push_back(decode_fst);
        }
        std::vector< std::vector<float> > scores(keyword_ids.size());
        std::vector< std::vector< std::pair<int,int> > > areas(keyword_ids.size());
        AcousticKeywordSpotting akwspotter;
        for (; !feature_reader.Done(); feature_reader.Next(), num_done++) {
            // for each keyword we first get all it's transition id and 
            // then calculate a likehood matrix and pass it to ivd model
            std::string utt = feature_reader.Key();
            Matrix<BaseFloat> features(feature_reader.Value());
            feature_reader.FreeCurrent();
            if (features.NumRows() == 0) {
                KALDI_WARN << "Zero-length utterance: " << utt;
                num_fail++;
                continue;
            }

            DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, 
                                                   features, acoustic_scale);    
            for (int i = 0; i < keyword_ids.size(); i++) {
                std::pair<int,int> area;
                float score = akwspotter.SegmentationByFillerReestimation(0,
                            fsts[i], &gmm_decodable, area);
                scores[i].push_back(score);          
                areas[i].push_back(area);
            } 
        }
        for (int i = 0; i < keyword_ids.size(); i++) {
            std::ofstream out(result_dir + "/" + keyword_ids[i] + ".RESULT");
            for (int j = 0; j < scores[i].size(); j++) {
                out << scores[i][j] << " "
                    << areas[i][j].first << " " 
                    << areas[i][j].second << std::endl;
            }
            out.close();
       } 
    } catch (const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
