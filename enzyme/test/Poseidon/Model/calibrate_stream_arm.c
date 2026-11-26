// poseidon-calibrate's stream arm with stand-in tools (an AMD GPU): the
// harness gets the model's fmuladd rows, its six rows join the model and leave
// the others, are measured again only with --force, and --check lists them; a
// model without fmuladd rows, a harness that fails or writes some rows leaves
// the model as it was; a host has no stream arm.
// RUN: rm -rf %t && mkdir %t %t/bin %t/cm
// RUN: cp %poseidon_calibrate %t/bin/
// RUN: printf '#!/bin/sh\nprintf "AMD Instinct MI300A\\tgfx942\\n"\n' > %t/bin/poseidon-calibrate-ops-hip
// RUN: printf '#!/bin/sh\necho "$*" > %t/args\nprintf "stream_load,double,0.024\\nstream_load,float,0.012\\nstream_store,double,0.031\\nstream_store,float,0.018\\nstream_combine,pnorm,12\\nstream_combine,band,%%%%s\\n" "$BAND"\n' > %t/bin/poseidon-calibrate-stream-hip
// RUN: chmod +x %t/bin/poseidon-calibrate-ops-hip %t/bin/poseidon-calibrate-stream-hip
// RUN: sed 's/^# native_arch=sm_120$/# native_arch=gfx942/' %cost_models/cm_sm_120_synthetic.csv > %t/cm/m.csv
// RUN: cp %t/cm/m.csv %t/before.csv
// RUN: not env PATH=%t/bin %t/bin/poseidon-calibrate --gpu 0 --check --out %t/cm/m.csv | FileCheck --check-prefix=UNLISTED %s
// RUN: env PATH=%t/bin BAND=0.02 %t/bin/poseidon-calibrate --gpu 0 --only stream --out %t/cm/m.csv | FileCheck --check-prefix=FIRST %s
// RUN: FileCheck --check-prefix=ARGS %s < %t/args
// RUN: grep -v '^stream_' %t/cm/m.csv | diff %t/before.csv -
// RUN: not env PATH=%t/bin %t/bin/poseidon-calibrate --gpu 0 --check --out %t/cm/m.csv | FileCheck --check-prefix=LISTED %s
// RUN: env PATH=%t/bin BAND=0.03 %t/bin/poseidon-calibrate --gpu 0 --only stream --out %t/cm/m.csv | FileCheck --check-prefix=AGAIN %s
// RUN: env PATH=%t/bin BAND=0.03 %t/bin/poseidon-calibrate --gpu 0 --only stream --force --out %t/cm/m.csv | FileCheck --check-prefix=FORCE %s
// RUN: grep '^stream_' %t/cm/m.csv | FileCheck --check-prefix=ROWS %s
// RUN: grep -v '^fmuladd,' %t/before.csv > %t/cm/nofma.csv
// RUN: not env PATH=%t/bin BAND=0.02 %t/bin/poseidon-calibrate --gpu 0 --only stream --out %t/cm/nofma.csv 2>&1 | FileCheck --check-prefix=NOFMA %s
// RUN: printf '#!/bin/sh\nprintf "stream_load,double,0.024\\nstream_store,double,0.031\\n"\n' > %t/bin/poseidon-calibrate-stream-hip
// RUN: cp %t/before.csv %t/cm/nofit.csv
// RUN: not env PATH=%t/bin %t/bin/poseidon-calibrate --gpu 0 --only stream --out %t/cm/nofit.csv 2>&1 | FileCheck --check-prefix=NOFIT %s
// RUN: diff %t/before.csv %t/cm/nofit.csv
// RUN: printf '#!/bin/sh\necho no memory >&2\nexit 1\n' > %t/bin/poseidon-calibrate-stream-hip
// RUN: not env PATH=%t/bin %t/bin/poseidon-calibrate --gpu 0 --only stream --out %t/cm/nofit.csv 2>&1 | FileCheck --check-prefix=FAILED %s
// RUN: diff %t/before.csv %t/cm/nofit.csv
// RUN: not %poseidon_calibrate --cpu x86-64 --only stream --out %t/cm/host.csv 2>&1 | FileCheck --check-prefix=HOST %s
// RUN: test ! -e %t/cm/host.csv

// UNLISTED: MISSING     stream_load  (streamed memory traffic){{$}}

// FIRST: poseidon-calibrate: device MI300A (gfx942, id 0)
// FIRST-NEXT: poseidon-calibrate: measuring streamed memory traffic
// FIRST-NEXT:   stream_load,double,0.024
// FIRST-NEXT:   stream_load,float,0.012
// FIRST-NEXT:   stream_store,double,0.031
// FIRST-NEXT:   stream_store,float,0.018
// FIRST-NEXT:   stream_combine,pnorm,12
// FIRST-NEXT:   stream_combine,band,0.02
// FIRST-NEXT: poseidon-calibrate: cost model: {{.*}}/cm/m.csv{{$}}

// ARGS: {{^}}512 1{{$}}

// LISTED: have        stream_load  (streamed memory traffic){{$}}

// AGAIN: poseidon-calibrate: device MI300A (gfx942, id 0)
// AGAIN-NEXT: poseidon-calibrate: stream: already measured
// AGAIN-NEXT: poseidon-calibrate: cost model: {{.*}}/cm/m.csv{{$}}

// FORCE: poseidon-calibrate: device MI300A (gfx942, id 0)
// FORCE-NEXT: poseidon-calibrate: measuring streamed memory traffic
// FORCE-NEXT:   stream_load,double,0.024
// FORCE-NEXT:   stream_load,float,0.012
// FORCE-NEXT:   stream_store,double,0.031
// FORCE-NEXT:   stream_store,float,0.018
// FORCE-NEXT:   stream_combine,pnorm,12
// FORCE-NEXT:   stream_combine,band,0.03
// FORCE-NEXT: poseidon-calibrate: cost model: {{.*}}/cm/m.csv{{$}}

// ROWS: {{^}}stream_load,double,0.024{{$}}
// ROWS-NEXT: {{^}}stream_load,float,0.012{{$}}
// ROWS-NEXT: {{^}}stream_store,double,0.031{{$}}
// ROWS-NEXT: {{^}}stream_store,float,0.018{{$}}
// ROWS-NEXT: {{^}}stream_combine,pnorm,12{{$}}
// ROWS-NEXT: {{^}}stream_combine,band,0.03{{$}}
// ROWS-NOT: stream_

// NOFMA: poseidon-calibrate: {{.*}}/cm/nofma.csv has no fmuladd,double and fmuladd,float rows, which the stream arm fits against; run the ops arm first{{$}}

// NOFIT: poseidon-calibrate: the stream calibration produced no stream_load,float, stream_store,float, stream_combine,pnorm, stream_combine,band row, so no stream row is written{{$}}
// NOFIT-NOT: poseidon-calibrate: cost model:

// FAILED: poseidon-calibrate: the stream calibration failed{{$}}
// FAILED-NOT: poseidon-calibrate: cost model:

// HOST: poseidon-calibrate: the stream arm measures a GPU, and --cpu a host{{$}}
