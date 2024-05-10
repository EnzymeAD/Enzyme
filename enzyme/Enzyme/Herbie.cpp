#include "Herbie.h"

#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Program.h>

#include <fstream>
#include <string>

void runViaHerbie(const std::string &cmd) {
  std::string tmpin = "/tmp/herbie_input";
  std::string tmpout = "/tmp/herbie_output";

  std::ofstream input(tmpin);
  if (!input) {
    llvm::errs() << "Failed to open input file.\n";
    return;
  }
  input << cmd;
  input.close();

  const char *Program = HERBIE_BINARY;
  llvm::StringRef Args[] = {"shell"};
  llvm::ArrayRef<llvm::Optional<llvm::StringRef>> Redirects = {
      llvm::StringRef(tmpin),  // stdin
      llvm::StringRef(tmpout), // stdout
      llvm::StringRef(tmpout)  // stderr
  };

  std::string ErrMsg;
  bool ExecutionFailed = false;

  llvm::sys::ExecuteAndWait(Program, Args, /*Env=*/llvm::None,
                            /*Redirects=*/Redirects,
                            /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg,
                            &ExecutionFailed);

  if (ExecutionFailed) {
    llvm::errs() << "Execution failed: " << ErrMsg << "\n";
    return;
  }

  std::ifstream output(tmpout);
  if (!output) {
    llvm::errs() << "Failed to open output file.\n";
    return;
  }

  std::string line;
  while (std::getline(output, line)) {
    llvm::errs() << line << "\n";
  }
  output.close();
}
