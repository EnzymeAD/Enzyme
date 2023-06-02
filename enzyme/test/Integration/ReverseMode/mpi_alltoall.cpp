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

void alltoall(float* sendbuf, int count, float* recvbuf) {
    MPI_Alltoall(sendbuf, count, MPI_FLOAT, recvbuf, count, MPI_FLOAT, MPI_COMM_WORLD);
}

void alltoall_adjoint(float* sendbuf, float* dsendbuf, int count, float* recvbuf, float* drecvbuf) {

    // Primal
    MPI_Alltoall(sendbuf, count, MPI_FLOAT, recvbuf, count, MPI_FLOAT, MPI_COMM_WORLD);

    // Adjoint
    std::vector<float> dsendbuf_tmp(count * n_procs);
    MPI_Alltoall(drecvbuf, count, MPI_FLOAT, dsendbuf_tmp.data(), count, MPI_FLOAT, MPI_COMM_WORLD);
    std::fill(drecvbuf, drecvbuf + (count * n_procs), 0);
    for(int i = 0; i < count * n_procs; ++i) {
        dsendbuf[i] += dsendbuf_tmp[i];
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
    int block = 2;
    int bufsize = n_procs * block;
    std::vector<float> sendbuf(bufsize, proc_id + 1);
    std::vector<float> recvbuf(bufsize, 0);
    std::vector<float> dsendbuf(bufsize, 0);
    std::vector<float> drecvbuf(bufsize, 0);
    if (proc_id == 0) {
        drecvbuf[bufsize - 1] = 1;
    }
    return std::make_tuple(sendbuf, dsendbuf, block, recvbuf, drecvbuf);
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
        auto [sendbuf, dsendbuf, block, recvbuf, drecvbuf] = getTestData(n_procs, proc_id);

        printAllOnRankZero(sendbuf, "send");

        alltoall(sendbuf.data(), block, recvbuf.data());

        printAllOnRankZero(recvbuf, "recv");
    }

    auto [sendbuf_a, dsendbuf_a, block_a, recvbuf_a, drecvbuf_a] = getTestData(n_procs, proc_id);
    {
        // Execute adjoint-enzyme
        printAllOnRankZero(drecvbuf_a, "drecv");

        __enzyme_autodiff((void*)alltoall, sendbuf_a.data(), dsendbuf_a.data(), block_a, recvbuf_a.data(), drecvbuf_a.data());

        printAllOnRankZero(dsendbuf_a, "dsend");
    }

    auto [sendbuf_b, dsendbuf_b, block_b, recvbuf_b, drecvbuf_b] = getTestData(n_procs, proc_id);
    {
        // Execute adjoint-manual
        printAllOnRankZero(drecvbuf_b, "drecv");

        alltoall_adjoint(sendbuf_b.data(), dsendbuf_b.data(), block_b, recvbuf_b.data(), drecvbuf_b.data());

        printAllOnRankZero(dsendbuf_b, "dsend");
    }

    {
        // Test
        bool testResult = AllCheckApproxEqual(sendbuf_a, sendbuf_b);
        testResult =  AllCheckApproxEqual(recvbuf_a, recvbuf_b);
        testResult =  AllCheckApproxEqual(dsendbuf_a, dsendbuf_b);
        testResult =  AllCheckApproxEqual(drecvbuf_a, drecvbuf_b);
        if(proc_id == 0) {
            std::cout << "Test " << (testResult ? "passed" : "failed") << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
