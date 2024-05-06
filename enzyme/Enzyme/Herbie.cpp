

void runViaHerbie(std::string cmd) {

    auto Program = "/path/to/herbie";
    // stdin for reading from, and stdout for writing from
    // todo write input to tmpin
    
    StringRef Args[] = {"shell", tmpin, tmpout};

    std::string ErrMsg;
    bool ExecutionFailed = false;
    llvm::sys::ExecuteAndWait(Program, Args, /*Env*/std::nullopt,
	{},
/*SecondsToWait*/0,
/*MemoryLimit */0, &ErrMsg,
&ExecutionFailed = nullptr);


    // parse output from tmpout


    return result;


}