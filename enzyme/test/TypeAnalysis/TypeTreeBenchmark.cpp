#include "Enzyme/TypeAnalysis/TypeTree.h"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <map>
#include <random>
#include <type_traits>
#include <vector>

using OldConcreteTypeMapType = std::map<std::vector<int>, ConcreteType>;

static std::vector<std::vector<int>> makeKeys(size_t Count) {
  std::vector<std::vector<int>> Keys;
  Keys.reserve(Count);
  for (size_t I = 0; I < Count; ++I)
    Keys.push_back({static_cast<int>(I / 8), static_cast<int>(I % 8),
                    static_cast<int>(I % 3)});

  std::mt19937 Generator(0);
  std::shuffle(Keys.begin(), Keys.end(), Generator);
  return Keys;
}

static ConcreteType integerType() { return ConcreteType(BaseType::Integer); }

template <typename MapType>
static MapType makeMap(const std::vector<std::vector<int>> &Keys) {
  MapType Map;
  for (const auto &Key : Keys)
    Map.emplace(Key, integerType());
  return Map;
}

template <typename MapType>
static void BM_Insert(benchmark::State &State) {
  const auto Keys = makeKeys(State.range(0));
  for (auto _ : State) {
    MapType Map;
    for (const auto &Key : Keys)
      benchmark::DoNotOptimize(Map.emplace(Key, integerType()));
    benchmark::ClobberMemory();
  }
}

template <typename MapType>
static void BM_Find(benchmark::State &State) {
  const auto Keys = makeKeys(State.range(0));
  const auto Map = makeMap<MapType>(Keys);
  size_t Index = 0;
  for (auto _ : State) {
    const auto Found = Map.find(Keys[Index++ % Keys.size()]);
    benchmark::DoNotOptimize(Found != Map.end());
  }
}

template <typename MapType>
static void BM_FindMissing(benchmark::State &State) {
  const auto Keys = makeKeys(State.range(0));
  const auto Map = makeMap<MapType>(Keys);
  const std::vector<int> MissingKey = {-2, -2, -2};
  for (auto _ : State) {
    const auto Found = Map.find(MissingKey);
    benchmark::DoNotOptimize(Found != Map.end());
  }
}

template <typename MapType>
static void BM_LexicographicTraversal(benchmark::State &State) {
  const auto Keys = makeKeys(State.range(0));
  const auto Map = makeMap<MapType>(Keys);
  for (auto _ : State) {
    size_t Count = 0;
    if constexpr (std::is_same_v<MapType, OldConcreteTypeMapType>) {
      for (const auto &Entry : Map)
        benchmark::DoNotOptimize(Count += Entry.first.size());
    } else {
      for (auto It = Map.lexicographic_begin(),
                End = Map.lexicographic_end();
           It != End; ++It)
        benchmark::DoNotOptimize(Count += It->first.size());
    }
    benchmark::ClobberMemory();
  }
}

template <typename MapType>
static void BM_LexicographicCompare(benchmark::State &State) {
  const auto Keys = makeKeys(State.range(0));
  const auto LHS = makeMap<MapType>(Keys);
  const auto RHS = makeMap<MapType>(Keys);
  for (auto _ : State)
    benchmark::DoNotOptimize(LHS < RHS);
}

template <typename MapType>
static void BM_Erase(benchmark::State &State) {
  const auto Keys = makeKeys(State.range(0));
  for (auto _ : State) {
    auto Map = makeMap<MapType>(Keys);
    for (const auto &Key : Keys)
      benchmark::DoNotOptimize(Map.erase(Key));
    benchmark::ClobberMemory();
  }
}

static void BM_BatchErase(benchmark::State &State) {
  const auto Keys = makeKeys(State.range(0));
  for (auto _ : State) {
    auto Map = makeMap<ConcreteTypeMapType>(Keys);
    benchmark::DoNotOptimize(Map.erase(Keys));
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_Insert, OldConcreteTypeMapType)->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_Insert, ConcreteTypeMapType)->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_Find, OldConcreteTypeMapType)->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_Find, ConcreteTypeMapType)->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_FindMissing, OldConcreteTypeMapType)->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_FindMissing, ConcreteTypeMapType)->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_LexicographicTraversal, OldConcreteTypeMapType)
  ->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_LexicographicTraversal, ConcreteTypeMapType)
    ->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_LexicographicCompare, OldConcreteTypeMapType)
  ->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_LexicographicCompare, ConcreteTypeMapType)
  ->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_Erase, OldConcreteTypeMapType)->Range(64, 4096);
BENCHMARK_TEMPLATE(BM_Erase, ConcreteTypeMapType)->Range(64, 4096);
BENCHMARK(BM_BatchErase)->Range(64, 4096);

BENCHMARK_MAIN();
