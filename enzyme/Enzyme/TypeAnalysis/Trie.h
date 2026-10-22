class Trie : public std::enable_shared_from_this<Trie> {
private:
  std::map<int, Trie *> mapping;
  std::optional<ConcreteType> ct;

public:
  Trie(BaseType BT) : ct(BT) {};
  Trie() : Trie(ConcreteType(BaseType::Unknown)) {};
  Trie(ConcreteType dat) : ct(dat) {};

  /// Lookup the underlying ConcreteType at a given offset sequence
  /// or Unknown if none exists
  ConcreteType operator[](const std::vector<int> Seq) const {
    Trie *res;
    std::map<int, Trie *> Found0 = mapping;
    for (auto elem : Seq) {
      if (Found0.count(elem) == 1) {
        res = mapping.at(elem);
      } else if (Found0.count({-1}) == 1) {
        res = Found0.at({-1});
      } else {
        return BaseType::Unknown;
      }
      Found0 = res->mapping;
    }
    if (!res->ct.has_value())
      return BaseType::Unknown;
    return res->ct.value();
  }

  // Return true if this type tree is fully known (i.e. there
  // is no more information which could be added).
  bool IsFullyDetermined() const {
    Trie *res;
    std::map<int, Trie *> Found0 = mapping;
    while (1) {
      if ((res->ct.has_value()) && Found0.count({-1}) != 1) {
        // without -1 we can add random elements
        return false;
      }
      res = Found0.at({-1});
      Found0 = res->mapping;
      if (!res->has_value()) {
        if (res->ct.has_value()) {
          return (res->ct.value() != BaseType::Pointer);
        }
        // leaf type unknown:
        return false;
      }
    }
  }

  /// Prepend an offset to all mappings
  Trie Only(int Off, llvm::Instruction *orig) {
    // Trie Only(int Off, llvm::Instruction *orig) const {
    Trie res = Trie();
    auto entry = std::pair<int, Trie *>(Off, this);
    res.mapping.insert(entry);
    return res;
  }

  /// Peel off the outermost index at offset 0
  Trie Data0() const { return *this->mapping.at(0); }

  /// Return if changed
  bool insert(const std::vector<int> Seq, ConcreteType CT,
              bool intsAreLegalSubPointer = false) {
    size_t SeqSize = Seq.size();
    if (SeqSize > EnzymeMaxTypeDepth) {
      if (EnzymeTypeWarning) {
        if (CustomErrorHandler) {
          CustomErrorHandler("TypeAnalysisDepthLimit", nullptr,
                             ErrorType::TypeDepthExceeded, this, nullptr, nullptr);
        } else
          llvm::errs() << "not handling more than " << EnzymeMaxTypeDepth
          << " pointer lookups deep dt:" << str()
          << " adding v: " << to_string(Seq) << ": " << CT.str()
          << "\n";
      }
      return false;
    }
    if (SeqSize == 0) {
      ct = CT;
      // mapping.insert(std::pair<const std::vector<int>, ConcreteType>(Seq,
      // CT));
      return true;
    }

    // check types at lower pointer offsets are either pointer or
    // anything. Don't insert into an anything
    // TODO: check off by 1 for the for loop
    {
      if (ct.has_value()) {
        if (ct.value() == BaseType::Anything)
          return false;
        if (ct.value() != BaseType::Pointer) {
          llvm::errs() << "FAILED CT: " << str()
                       << " adding Seq: " << to_string(Seq) << ": " << CT.str()
                       << "\n";
        }
        assert(ct.value() == BaseType::Pointer);
      }
      std::vector<int> tmp(Seq);
      Trie *res;
      std::map<int, Trie *> Found0 = mapping;
      for (int i : Seq) {
        if (mapping.count(i) == 0)
          break;
        res = Found0.at(i);
        Found0 = res->mapping;

        if (res->ct.has_value()) {
          if (ct.value() == BaseType::Anything)
            return false;
          if (res->ct.value() != BaseType::Pointer) {
            llvm::errs() << "FAILED CT: " << str()
                         << " adding Seq: " << to_string(Seq) << ": "
                         << CT.str() << "\n";
          }
          assert(res->ct.value() == BaseType::Pointer);
        }
      }
    }

    bool changed = false;

    // if this is a ending -1, remove other elems if no more info
    if (Seq.back() == -1) {
      std::set<std::vector<int>> toremove;
      for (const auto &pair : mapping) {
        if (pair.first.size() != SeqSize)
          continue;
        bool matches = true;
        for (unsigned i = 0; i < SeqSize - 1; ++i) {
          if (pair.first[i] != Seq[i]) {
            matches = false;
            break;
          }
        }
        if (!matches)
          continue;

        if (intsAreLegalSubPointer && pair.second == BaseType::Integer &&
            CT == BaseType::Pointer) {
          toremove.insert(pair.first);
        } else {
          if (CT == pair.second) {
            // previous equivalent values or values overwritten by
            // an anything are removed
            toremove.insert(pair.first);
          } else if (pair.second != BaseType::Anything) {
            llvm::errs() << "inserting into : " << str() << " with "
                         << to_string(Seq) << " of " << CT.str() << "\n";
            llvm_unreachable("illegal insertion");
          }
        }
      }

      for (const auto &val : toremove) {
        mapping.erase(val);
        changed = true;
      }
    }

    // if this is a starting -1, remove other -1's
    if (Seq[0] == -1) {
      std::set<std::vector<int>> toremove;
      for (const auto &pair : mapping) {
        if (pair.first.size() != SeqSize)
          continue;
        bool matches = true;
        for (unsigned i = 1; i < SeqSize; ++i) {
          if (pair.first[i] != Seq[i]) {
            matches = false;
            break;
          }
        }
        if (!matches)
          continue;
        if (intsAreLegalSubPointer && pair.second == BaseType::Integer &&
            CT == BaseType::Pointer) {
          toremove.insert(pair.first);
        } else {
          if (CT == pair.second) {
            // previous equivalent values or values overwritten by
            // an anything are removed
            toremove.insert(pair.first);
          } else if (pair.second != BaseType::Anything) {
            llvm::errs() << "inserting into : " << str() << " with "
                         << to_string(Seq) << " of " << CT.str() << "\n";
            llvm_unreachable("illegal insertion");
          }
        }
      }

      for (const auto &val : toremove) {
        mapping.erase(val);
        changed = true;
      }
    }

    bool possibleDeletion = false;
    size_t minLen =
        (minIndices.size() <= SeqSize) ? minIndices.size() : SeqSize;
    for (size_t i = 0; i < SeqSize; i++) {
      if (minIndices[i] > Seq[i]) {
        if (minIndices[i] > MaxTypeOffset)
          possibleDeletion = true;
        minIndices[i] = Seq[i];
      }
    }

    if (possibleDeletion) {
      std::vector<std::vector<int>> toErase;
      for (const auto &pair : mapping) {
        size_t i = 0;
        bool considerErase = false;
        for (int val : pair.first) {
          if (val > MaxTypeOffset) {
            considerErase = true;
          }
          ++i;
        }
        if (considerErase) {
          toErase.push_back(pair.first);
        }
      }

      for (auto vec : toErase) {
        mapping.erase(vec);
        changed = true;
      }
    }

    size_t i = 0;
    bool keep = false;
    bool considerErase = false;
    for (auto val : Seq) {
      if (val > MaxTypeOffset) {
        if (val == minIndices[i]) {
          keep = true;
          break;
        }
        considerErase = true;
      }
      i++;
    }
    if (considerErase && !keep)
      return changed;
    mapping.insert(std::pair<const std::vector<int>, ConcreteType>(Seq, CT));
    return true;
  }
  
  bool operator<(const Trie &vd) const; // TODO;

  /// Whether this TypeTree contains any information
  bool isKnown() const {
    if (ct.has_value())
      return true;
    // The following might make it slower, still keep it?
    for (auto &&[idx, trie] : mapping) {
      bool inner = trie->isKnown();
      if (inner)
        return true;
    }
    return false;
  }

  /// Whether this TypeTree knows any non-pointer information
  bool isKnownPastPointer() const {
    for (auto &&[idx, trie] : mapping) {
      if (trie.ct.has_value()) {
        auto val = trie.ct.value();
        if (val != BaseType::Pointer && val != BaseType::Anything)
          return true;
      }
      bool inner = trie->isKnownPastPointer();
      if (inner)
        return true;
    }
    return false;
  }

  /// Select only the Integer ConcreteTypes
  TypeTree JustInt() const {
    TypeTree vd;
    for (auto &&[idx, trie] : mapping) {
      if (trie->ct.has_value() {
        auto val = trie->ct.value();
        if (val == BaseType::Integer) {
          vd.insert(idx, val);
        }
      }
    }
    return vd;
  }

  /// Returns a string representation of this TypeTree
  std::string str() const {
    std::string out = "{";
    bool first = true;
    for (auto &pair : mapping) {
      if (!first) {
        out += ", ";
      }
      out += "[";
      for (unsigned i = 0; i < pair.first.size(); ++i) {
        if (i != 0)
          out += ",";
        out += std::to_string(pair.first[i]);
      }
      out += "]:" + pair.second->str();
      first = false;
    }
    out += "}";
    return out;
  }
};
