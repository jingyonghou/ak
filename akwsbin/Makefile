
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compile-keyword-graphs iterating-viterbi-decoding

OBJFILES =



TESTFILES =

ADDLIBS = ../akws/kaldi-akws.a ../decoder/kaldi-decoder.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk

