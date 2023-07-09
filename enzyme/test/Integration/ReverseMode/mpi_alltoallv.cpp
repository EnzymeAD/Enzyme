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

extern void __enzyme_autodiff(void*, ...);

int proc_id, n_procs;

void alltoallv(float* sendbuf, int* sendcount, int* senddispl,
               float* recvbuf, int* recvcount, int* recvdispl) {
    MPI_Alltoallv(sendbuf, sendcount, senddispl, MPI_FLOAT, recvbuf, recvcount, recvdispl, MPI_FLOAT, MPI_COMM_WORLD);
}

void alltoallv_adjoint(float* sendbuf, float* dsendbuf, int* sendcount, int* senddispl,
                       float* recvbuf, float* drecvbuf, int* recvcount, int* recvdispl) {

    // Primal
    MPI_Alltoallv(sendbuf, sendcount, senddispl, MPI_FLOAT, recvbuf, recvcount, recvdispl, MPI_FLOAT, MPI_COMM_WORLD);

    // Adjoint
    int bufsize = std::accumulate(sendcount, sendcount + n_procs, 0);
    std::vector<float> dsendbuf_tmp(bufsize);
    MPI_Alltoallv(drecvbuf, recvcount, recvdispl, MPI_FLOAT, dsendbuf_tmp.data(), sendcount, senddispl, MPI_FLOAT, MPI_COMM_WORLD);
    std::fill(drecvbuf, drecvbuf + bufsize, 0);
    for(int i = 0; i < bufsize; ++i) {
        dsendbuf[i] += dsendbuf_tmp[i];
    }
}

void printAllOnRankZeroV(const std::vector<float>& buf, std::string name = "") {

    int bufsize = buf.size();
    std::vector<int> bufsizes(n_procs);
    MPI_Gather(&bufsize, 1, MPI_INT, bufsizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> bufoffsets(n_procs, 0);
    std::partial_sum(bufsizes.begin(), bufsizes.end()-1, bufoffsets.begin()+1);

    std::vector<float> globalbuf(std::accumulate(bufsizes.begin(), bufsizes.end(), 0));

    MPI_Gatherv(buf.data(), buf.size(), MPI_FLOAT, globalbuf.data(), bufsizes.data(), bufoffsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(proc_id == 0) {
        std::cout << name << "\n";
        for (int i = 0; i < n_procs; ++i) {
            int begin = bufoffsets[i];
            int end = bufoffsets[i] + bufsizes[i];
            if ((end - begin) <= 0) {
                return;
            }
            for (int i = begin; i < end - 1; ++i) {
                std::cout << globalbuf[i] << ",";
            }
            std::cout << globalbuf[end - 1] << "\n";
        }
    }
}

auto getTestData(int n_procs, int proc_id) {
    std::vector<int> sendcounts(n_procs);
    for (int i = 0; i < n_procs; ++i) {
        if((i + proc_id) % 2) {
            sendcounts[i] = 1;
        } else {
            sendcounts[i] = 2;
        }
    }
    std::vector<int> senddispl(n_procs, 0);
    std::partial_sum(sendcounts.begin(), sendcounts.end()-1, senddispl.begin()+1);
    int bufsize = std::accumulate(sendcounts.begin(), sendcounts.end(), 0);
    auto recvcounts = sendcounts;
    auto recvdispl = senddispl;

    std::vector<float> sendbuf(bufsize, proc_id + 1);
    std::vector<float> recvbuf(bufsize, 0);
    std::vector<float> dsendbuf(bufsize, 0);
    std::vector<float> drecvbuf(bufsize, 0);
    if (proc_id == 0) {
        drecvbuf[bufsize-1] = 1;
    }
    return std::make_tuple(sendbuf, dsendbuf, sendcounts, senddispl, recvbuf, drecvbuf, recvcounts, recvdispl);
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
        auto [sendbuf, dsendbuf, sendcount, senddispl, recvbuf, drecvbuf, recvcount, recvdispl] = getTestData(n_procs, proc_id);

        printAllOnRankZeroV(sendbuf, "send");

        alltoallv(sendbuf.data(), sendcount.data(), senddispl.data(),
                recvbuf.data(), recvcount.data(), recvdispl.data());

        printAllOnRankZeroV(recvbuf, "recv");
    }

    auto [sendbuf_a, dsendbuf_a, sendcount_a, senddispl_a, recvbuf_a, drecvbuf_a, recvcount_a, recvdispl_a] = getTestData(n_procs, proc_id);
    {
        // Execute adjoint - enzyme

        printAllOnRankZeroV(drecvbuf_a, "drecv");

        __enzyme_autodiff((void*)alltoallv,
                sendbuf_a.data(), dsendbuf_a.data(), sendcount_a.data(), senddispl_a.data(),
                recvbuf_a.data(), drecvbuf_a.data(), recvcount_a.data(), recvdispl_a.data());

        printAllOnRankZeroV(dsendbuf_a, "dsend");
    }

    auto [sendbuf_b, dsendbuf_b, sendcount_b, senddispl_b, recvbuf_b, drecvbuf_b, recvcount_b, recvdispl_b] = getTestData(n_procs, proc_id);
    {
        // Execute adjoint - manual

        printAllOnRankZeroV(drecvbuf_b, "drecv");

        alltoallv_adjoint(
                sendbuf_b.data(), dsendbuf_b.data(), sendcount_b.data(), senddispl_b.data(),
                recvbuf_b.data(), drecvbuf_b.data(), recvcount_b.data(), recvdispl_b.data());

        printAllOnRankZeroV(dsendbuf_b, "dsend");
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
