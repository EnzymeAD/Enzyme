// RUN: %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme --enzyme-inline=1 -S | %lli -

#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

// XFAIL: *

extern void __enzyme_fwddiff(void*, ...);

int proc_id, n_procs;

void allreducemax(float* sendbuf, float* recvbuf, int count) {
    MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
}

struct FloatInt {
    float value;
    int location;
};

void allreducemax_tangent(float* sendbuf, float* dsendbuf, float* recvbuf, float* drecvbuf, int count) {

    // Primal
    std::vector<int> recvlocbuf(count);
    auto mpi_allreduce_comploc_float = [](float* sendbuf, float* recvbuf, int* recvlocbuf, int count, MPI_Op op, MPI_Comm comm) {
        std::vector<FloatInt> sendbuf_tmp(count);
        std::vector<FloatInt> recvbuf_tmp(count);

        for(int i = 0; i < count; ++i) {
            sendbuf_tmp[i].value = sendbuf[i];
            sendbuf_tmp[i].location = proc_id;
        }

        MPI_Allreduce(sendbuf_tmp.data(), recvbuf_tmp.data(), count, MPI_FLOAT_INT, op, comm);

        for(int i = 0; i < count; ++i) {
            recvbuf[i] = recvbuf_tmp[i].value;
            recvlocbuf[i] = recvbuf_tmp[i].location;
        }
    };
    mpi_allreduce_comploc_float(sendbuf, recvbuf, recvlocbuf.data(), count, MPI_MAXLOC, MPI_COMM_WORLD);

    // Tangent
    {
        std::vector<float> dsendbuf_tmp(count);

        for(int i = 0; i < count; ++i) {
            dsendbuf_tmp[i] = (recvlocbuf[i] == proc_id) ? dsendbuf[i] : 0;
        }

        MPI_Allreduce(dsendbuf_tmp.data(), drecvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
}

void printAllOnRankZero(const std::vector<float>& buf, std::string name = "") {
    std::vector<float> globalbuf;
    if(proc_id == 0){
        std::cout << name << "\n";
        globalbuf.resize(buf.size() * n_procs);
    }
    MPI_Gather(buf.data(), buf.size(), MPI_FLOAT, globalbuf.data(), buf.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(proc_id == 0) {
        for(int i = 0; i < n_procs; ++i) {
            int begin = i*buf.size();
            int end = (i+1)*buf.size();
            for (int i = begin; i < end - 1; ++i) {
                std::cout << globalbuf[i] << ",";
            }
            std::cout << globalbuf[end - 1] << "\n";
        }
    }
}

auto getTestData(int n_procs, int proc_id) {
    int count = 2;
    std::vector<float> sendbuf(count, proc_id + 1);
    std::vector<float> recvbuf(count, 0);
    std::vector<float> dsendbuf(count, 0);
    std::vector<float> drecvbuf(count, 0);
    if (proc_id == (n_procs - 1)) {
        drecvbuf[count - 1] = 1;
    }
    return std::make_tuple(sendbuf, dsendbuf, recvbuf, drecvbuf, count);
}

bool AllCheckApproxEqual(const std::vector<float>& a, const std::vector<float>& b) {
    bool equal = true;
    if(a.size() != b.size()) {
        equal = false;
    } else {
        for(int i = 0; i < a.size(); ++i) {
            if(a[i] != b[i]){
                equal = false;
                break;
            }
        }
    }
    bool globalEqual;
    MPI_Allreduce(&equal, &globalEqual, 1, MPI_CHAR, MPI_LAND, MPI_COMM_WORLD);
    return globalEqual;
}

int main( int argc, char *argv[] )
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    {
        // Execute primal
        auto [sendbuf, dsendbuf, recvbuf, drecvbuf, count] = getTestData(n_procs, proc_id);

        printAllOnRankZero(sendbuf, "send");

        allreducemax(sendbuf.data(), recvbuf.data(), count);

        printAllOnRankZero(recvbuf, "recv");
    }

    auto [sendbuf_a, dsendbuf_a, recvbuf_a, drecvbuf_a, count_a] = getTestData(n_procs, proc_id);
    {
        // Execute adjoint-enzyme
        printAllOnRankZero(drecvbuf_a, "drecv");

        __enzyme_fwddiff((void*)allreducemax, sendbuf_a.data(), dsendbuf_a.data(), recvbuf_a.data(), drecvbuf_a.data(), count_a);

        printAllOnRankZero(dsendbuf_a, "dsend");
    }

    auto [sendbuf_b, dsendbuf_b, recvbuf_b, drecvbuf_b, count_b] = getTestData(n_procs, proc_id);
    {
        // Execute adjoint-manual
        printAllOnRankZero(drecvbuf_b, "drecv");

        allreducemax_tangent(sendbuf_b.data(), dsendbuf_b.data(), recvbuf_b.data(), drecvbuf_b.data(), count_b);

        printAllOnRankZero(dsendbuf_b, "dsend");
    }

    {
        // Test
        bool testResult = AllCheckApproxEqual(sendbuf_a, sendbuf_b);
        testResult = testResult && AllCheckApproxEqual(recvbuf_a, recvbuf_b);
        testResult = testResult && AllCheckApproxEqual(dsendbuf_a, dsendbuf_b);
        testResult = testResult && AllCheckApproxEqual(drecvbuf_a, drecvbuf_b);
        if(proc_id == 0) {
            std::cout << "Tests " << (testResult ? "passed" : "failed") << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
