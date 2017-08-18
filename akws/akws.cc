// alws/akws.cc

//Copyright 2017 MSRA(Author: jyhou@nwpu-aslp.org)

#include "akws/akws.h"
namespace kaldi {
namespace akws {
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using fst::ArcIterator;

fst::VectorFst<fst::StdArc> * AcousticKeywordSpotting::InverseArcOfFst(const fst::VectorFst<fst::StdArc> &fst) {
    VectorFst<StdArc> *ans = new VectorFst<StdArc>;
    state_num = fst.NumStates();
    for (StateId s = 0; s < state_num; s++) {
        ans->AddState();
    } 
    StdArc arc_start(0, 0, fst::Weight::One(), 0);
    ans->AddArc(0, arc_start);
    StdArc arc_end(0, 0, fst::Weight::One(), state_num-1);
    ans->AddArc(state_num-1, arc_end);
    // TODO here we should consider add filler at middle of key phrase 
    for (StateId s = 0; s < state_num; s++) {
        for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, s);
            !aiter.Done();
            aiter.Next()) {
            const StdArc &arc = aiter.Value();
            Label olabel = 0;
          StdArc newarc(arc.ilabel, olabel, arc.weight, s);
          ans->AddArc(arc.nextstate, newarc);
        }
    }
    return ans;
}


float AcousticKeywordSpotting::VertibeDecode(const fst::VectorFst<fst::StdArc> &fst, Matrix<BaseFloat> &acoustic_cost) {
    size_t state_num = acoustic_cost.NumRows();
    size_t frame_num = acoustic_cost.NumCols();

    Matrix<BaseFloat> cost(state_num, frame_num);
    Matrix<int> path(state_num, frame_num);
    Matrix<int> s_point(state_num, frame_num);

    cost(0, 0) = acoustic_cost(0, 0);
    last_state(0, 0) = -1;
    s_point(0, 0) = 0;

    int i;
    double cost_0, co
    for (i = 1; i < frame_num; i++) {
        cost(0, i) = cost(0, i-1) + acoustic_cost(0, i);
        last_state(0, i) = 0;
        s_point(0, i) = i    
    }

    for (s = 1; s < state_num; s++) {
        cost(s, 0) = BIG_FLT; 
        last_state(s, 0) = -1;
        s_point(s, 0) = -1;
    }

    for (i = 1; i < frame_num; i++) {
        for (s = 1; s < state_num; s++) {
            float best_cost = 2*BIG_FLT;
            int best_last_state = -1;
            int best_s_point = -1;        
            for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, s);
                !aiter.Done();
                aiter.Next()) {
                const StdArc &arc = aiter.Value();
                if (best_cost > cost(arc.nextstate, i-1)+arc.weight) {
                    best_cost = cost(arc.nextstate, i-1)+arc.weight;
                    best_last_state = arc.nextstate;
                    best_s_point = s_point(arc.nextstate, i-1)
                }
            }
            cost(s, i) = best_cost;
            last_state(s, i) = best_last_state;
            s_point(s, 0) = best_s_point;
        }

    }
    
    s = state_num - 1;
    i = frame_num - 1;
    while(last_state(s, i) == s) {
        i--;
    }
    return (cost(s, i) - cost(0, s_point(s, i)) )/(i-s_point(s, i)-1);
}

float AcousticKeywordSpotting::SegmentationByFillerReestimation(const fst::VectorFst<fst::StdArc> &fst, DecodableInterface *decodable){
    typedef typename StdArc::Weight Weight;
    typedef typename StdArc::StateId StateId;
    typedef typename StdArc::Label Label;

    float epsilon = 0;
    size_t state_num = fst.NumStates();
    size_t frame_num = decodable->NumFramesReady();
    Matrix<BaseFloat> loglikehoods(state_num, frame_num);

    size_t i;
    StateId s;
    BaseFloat acoustic_cost;

    for (i = 0; i < frame_num; i++) {
        for (s = 0; s < state_num; s++) {
            acoustic_cost = epsilon;
            for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, s);
                !aiter.Done();
                aiter.Next()) { 
                const StdArc &arc = aiter.Value();
                if (arc.ilabel != 0) {
                    acoustic_cost = -decodable->LogLikelihood(i, arc.ilabel);
                    break;
                }
            }
            loglikehoods(s, i) = acoustic_cost;
        }
    }
    VectorFst<StdArc> *i_fst = InverseArcOfFst(fst);
    return 0;    
}
} //namespace akws
} //namespace kaldi
