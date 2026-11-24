// RUN: %eopt --sicm='rule-options=novalue,unknown=1' --verify-diagnostics %s

// expected-error @+2 {{sicm: rule-options entry 'novalue' is not of the form key=value}}
// expected-error @+1 {{sicm: no loaded rule set has an option 'unknown'}}
module {
}
