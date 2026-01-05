include "Common.td"

def : InactiveOp<"enzyme", "IgnoreDerivativesOp">;

def : InactiveOp<"enzyme", "InitTraceOp">;
def : InactiveOp<"enzyme", "AddWeightToTraceOp">;
def : InactiveOp<"enzyme", "AddSampleToTraceOp">;
def : InactiveOp<"enzyme", "AddSubtraceOp">;
def : InactiveOp<"enzyme", "AddRetvalToTraceOp">;
def : InactiveOp<"enzyme", "GetSampleFromTraceOp">;
def : InactiveOp<"enzyme", "GetSubtraceOp">;
def : InactiveOp<"enzyme", "GetWeightFromTraceOp">;
def : InactiveOp<"enzyme", "GetFlattenedSamplesFromTraceOp">;
def : InactiveOp<"enzyme", "GetSampleFromConstraintOp">;
def : InactiveOp<"enzyme", "GetSubconstraintOp">;
