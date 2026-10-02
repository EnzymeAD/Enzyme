; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme-summary" -disable-output | FileCheck %s; fi

; Registrations written in C may declare the registered Fortran procedure as
; a variable ("extern char sym;"); the summary still records them.

@_QMmo_exceptionPmessage = external global i8
@__enzyme_inactivefn_message = global ptr @_QMmo_exceptionPmessage
@__enzyme_nofree_message = global ptr @_QMmo_exceptionPmessage

; CHECK: "registrations": {
; CHECK-DAG: "inactive": [
; CHECK-DAG: "nofree": [
; CHECK-DAG: "_QMmo_exceptionPmessage"
