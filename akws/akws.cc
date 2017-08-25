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

fst::VectorFst<fst::StdArc> *AcousticKeywordSpotting::InverseArcOfFst(const fst::VectorFst<fst::StdArc> &fst) {
    typedef typename StdArc::Weight Weight;
    typedef typename StdArc::StateId StateId;
    typedef typename StdArc::Label Label;

    VectorFst<StdArc> *ans = new VectorFst<StdArc>;
    StateId state_num = fst.NumStates();
    for (StateId s = 0; s < state_num; s++) {
        ans->AddState();
    } 
    StdArc arc_start(0, 0, Weight::One(), 0);
    ans->AddArc(0, arc_start);
    StdArc arc_end(0, 0, Weight::One(), state_num-1);
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

float AcousticKeywordSpotting::VertibeDecode(const fst::VectorFst<fst::StdArc> &fst, 
            Matrix<BaseFloat> &acoustic_cost, std::pair<int, int> &area) {
    typedef typename StdArc::Weight Weight;
    typedef typename StdArc::StateId StateId;
    typedef typename StdArc::Label Label;
    
    StateId state_num = acoustic_cost.NumRows();
    MatrixIndexT frame_num = acoustic_cost.NumCols();

    Matrix<BaseFloat> cost(state_num, frame_num);
    Matrix<BaseFloat> last_state(state_num, frame_num);
    Matrix<BaseFloat> s_point(state_num, frame_num);

    cost(0, 0) = acoustic_cost(0, 0);
    last_state(0, 0) = -1;
    s_point(0, 0) = 0;

    MatrixIndexT i;
    StateId s;
    for (i = 1; i < frame_num; i++) {
        cost(0, i) = cost(0, i-1) + acoustic_cost(0, i);
        last_state(0, i) = 0;
        s_point(0, i) = i; 
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
                if (best_cost > ((cost(arc.nextstate, i-1) + arc.weight.Value()))) {
                    best_cost = cost(arc.nextstate, i-1) + arc.weight.Value();
                    best_last_state = arc.nextstate;
                    best_s_point = s_point(arc.nextstate, i-1);
                }
            }
            cost(s, i) = best_cost + acoustic_cost(s, i);
            last_state(s, i) = best_last_state;
            s_point(s, i) = best_s_point;
        }

    }
    
    //back trace the spotting area
    s = state_num - 1;
    i = frame_num - 1;
    while(int(last_state(s, i)) == s) {
        i--;
    }

    // get the last transition probability 
    float last_weight;
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, s); 
         !aiter.Done(); aiter.Next()) {
        const StdArc &arc = aiter.Value();
        if (arc.nextstate == last_state(s, i)) {
            last_weight = arc.weight.Value();
        }
    }
    area.first = int(s_point(s, i)) + 1;
    area.second = i - 1;
    return (cost(s, i) - cost(0, area.first-1) - acoustic_cost(0,0))/(area.second-area.first+1);
}

float AcousticKeywordSpotting::SegmentationByFillerReestimation(float epsilon, 
    const fst::VectorFst<fst::StdArc> &fst, 
    DecodableInterface *decodable, 
    std::pair<int, int> &area) {

    typedef typename StdArc::Weight Weight;
    typedef typename StdArc::StateId StateId;
    typedef typename StdArc::Label Label;

    StateId state_num = fst.NumStates();
    MatrixIndexT frame_num = decodable->NumFramesReady();
    Matrix<BaseFloat> loglikehoods(state_num, frame_num);

    MatrixIndexT i;
    StateId s;
    BaseFloat acoustic_cost;
    float best_epsilon = 1;
    int counter = 1;

    std::vector<StateId> nonemittion_states;
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
    std::ofstream out("distance.txt");
    for (int p = 0; p < state_num; p++) {
        for (int q = 0; q < frame_num; q++) {
           out << loglikehoods(p, q)  << " ";
        }
        out << std::endl;
    }
    out.close();

    for (s = 0; s < state_num; s++) {
        int flag = 0;
        for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, s); 
             !aiter.Done(); aiter.Next()) { 
            const StdArc &arc = aiter.Value();
            if (arc.ilabel != 0) {
                flag = 1;
                break;
            }
        }
        if (flag != 1) {
            nonemittion_states.push_back(s);
        }
    }

    VectorFst<StdArc> *i_fst = InverseArcOfFst(fst);
    while ((best_epsilon - epsilon) > 1e-3f || best_epsilon < epsilon) {
        best_epsilon = epsilon;
        epsilon = VertibeDecode(*i_fst, loglikehoods, area);
        /*std::cout << "iteration: " << counter << ", epsilon: " << epsilon 
            << ", start frame: " << area.first << ", end frame: " << area.second << std::endl;
        */
        /*if (epsilon > 1) {
            std::cout << "too short utterances: (" << frame_num  << " vs "
                      << state_num << ")" << std::endl;
            break;
        }*/

        counter++;
        if (counter > MAX_ITERATION) {
            std::cout << "exceed maximum iterations: " << counter << std::endl;
            std::cout << "precision: " << best_epsilon - epsilon << std::endl;
            break;
        }

        // here we reinit the epsilon for nonemittion states
        for (i = 0; i < frame_num; i++) {
            for (std::vector<StateId>::const_iterator siter=nonemittion_states.begin(); 
            siter != nonemittion_states.end(); siter++) {
                loglikehoods(*siter, i) = epsilon;
            }
        } 
    }
    return best_epsilon;    
}
} //namespace akws
} //namespace kaldi
