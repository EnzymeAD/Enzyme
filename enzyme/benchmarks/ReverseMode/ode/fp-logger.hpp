void initializeLogger();
void destroyLogger();
void printLogger();

void enzymeLogError(const char *id, double err);
void enzymeLogGrad(const char *id, double grad);
void enzymeLogValue(const char *id, double res, unsigned numOperands,
                    double *operands);
