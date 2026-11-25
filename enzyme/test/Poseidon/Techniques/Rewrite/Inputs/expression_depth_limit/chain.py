# Writes @tester, a chain of 102 dependent additions (one deeper than the
# deepest expression Poseidon forms), to argv[1] and its profile to argv[2].
import sys

LENGTH = 102
CANONICAL_HASH = "c38c6fa90a84343a"

module_path, profile_path = sys.argv[1:]
with open(module_path, "w") as module:
    module.write("define double @tester(double %x, double %y) #0 {\nentry:\n")
    module.write("  %a0 = fadd double %x, %y\n")
    for i in range(1, LENGTH):
        module.write(f"  %a{i} = fadd double %a{i - 1}, %y\n")
    module.write(f"  ret double %a{LENGTH - 1}\n}}\n")
with open(profile_path, "w") as profile:
    profile.write(f"CanonicalHash = {CANONICAL_HASH}\n")
    for i in range(LENGTH):
        profile.write(
            f"{i}\n\tMinRes = 1.0\n\tMaxRes = 4.0\n"
            "\tSumSens = 100.0\n\tSumGrad = 100.0\n\tSumAbsGrad = 100.0\n"
            "\tExec = 100\n\tNumOperands = 2\n"
            "\tOperand[0] = [1.0, 2.0, 0.0]\n"
            "\tOperand[1] = [1.0, 2.0, 0.0]\n")
