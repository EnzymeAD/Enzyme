// The driver reads its arguments after expanding response files, as clang
// does: a Poseidon flag in one reaches the pass, and a response file that holds
// only flags is no input, so a line without one gets the user's arguments
// alone. A command too long for the system goes to clang in a response file of
// the driver's own, which is removed after the compile.
//
// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t/tmp
// RUN: printf -- '-poseidon-profile-use -poseidon-tau=1e-7\n' > %t/flags.rsp
// RUN: printf -- '@%t/flags.rsp -poseidon-cache=%t/cache\n' > %t/nested.rsp
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx @%t/nested.rsp -x c -### %s \
// RUN:     -o %t/a.out 2>&1 | FileCheck --check-prefix=FLAGS %s
// RUN: printf -- '-v\n' > %t/version.rsp
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx @%t/version.rsp 2>&1 \
// RUN:   | FileCheck --check-prefix=VERSION %s
// RUN: %python -c "print('-DLONG=' + 'a' * 140000)" > %t/long.rsp
// RUN: env TMPDIR=%t/tmp %poseidon_clang -Werror -fsyntax-only -DEXPECT_LONG \
// RUN:     -x c @%t/long.rsp %s
// RUN: ls %t/tmp | count 0

#if defined(EXPECT_LONG) && !defined(LONG)
#error "the macro of the response file did not arrive"
#endif
int main(void) { return 0; }

// FLAGS: clang{{.*}} -fpass-plugin={{.*}}Poseidon-{{[0-9]+}}.so
// FLAGS-SAME: -Xclang -mllvm -Xclang -poseidon-profile-use
// FLAGS-SAME: -Xclang -mllvm -Xclang -poseidon-tau=1e-7
// FLAGS-SAME: -Xclang -mllvm -Xclang -poseidon-cache=
// FLAGS-NOT: unknown argument

// VERSION: clang++ -v{{ ?$}}
// VERSION-NEXT: clang version
// VERSION-NOT: "{{[^"]*}}ld{{(\.lld)?}}"
