// akws/akws.h

//Copyright 2017 MSRA (Author: jyhou@nwpu-aslp.org)

#ifndef KALDI_AKWS_AKWS_H_
#define KALDI_AKWS_AKWS_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "fst/fstlib.h"
#include "matrix/matrix-lib.h"
#include "itf/decodable-itf.h"

namespace kaldi {
namespace akws {

#define BIG_FLT 9999999
#define MAX_ITERATION 10 

class AcousticKeywordSpotting {

public:
    AcousticKeywordSpotting() {}

    ~AcousticKeywordSpotting() {}    

public:
    float SegmentationByFillerReestimation(const fst::VectorFst<fst::StdArc> &fst, DecodableInterface *decodable);

    fst::VectorFst<fst::StdArc> * InverseArcOfFst(const fst::VectorFst<fst::StdArc> &fst);
};

} //namespace akws
} //namespace kaldi
#endif
