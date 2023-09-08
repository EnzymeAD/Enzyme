#pragma once

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

#define private public

template <typename Key, typename T, typename Allocator> class trie;
template <typename Key, typename Index, typename T, typename Allocator>
class trie_const_iterator;
template <typename Key, typename Index, typename Value> struct trie_node;

// MARK: - Iterator

template <typename Key, typename Index, typename T, typename Allocator>
class trie_iterator {
private:
  using key_type = Key;
  using index_type = Index;
  using allocator_type = Allocator;
  using self = trie_iterator<Key, Index, T, Allocator>;

public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = std::pair<const key_type, value_type*>;
  using reference = std::pair<const key_type, value_type&>;

private:
  template <typename S, typename C, typename M, typename A>
  friend class trie_const_iterator;

  template <typename KeyT, typename TT, typename AllocatorT> friend class trie;

  trie<Key, T, Allocator> *_trie;
  trie_node<Key, Index, T> *_node;

public:
  trie_iterator() : _trie(nullptr), _node(nullptr) {}

  explicit trie_iterator(trie<Key, T, Allocator> *trie,
                         trie_node<Key, Index, value_type> *n)
      : _trie(trie), _node{n} {}

  trie_iterator(const trie_iterator &other)
      : _trie(other._trie), _node(other._node) {}

  trie_iterator &operator=(const trie_iterator &other) {
    if (this == &other) {
      return *this;
    }
    _trie = other._trie;
    _node = other._node;
    return *this;
  }

  reference operator*() const {
    assert(_node != nullptr && "dereferencing end() iterator");
    return std::make_pair(_node->get_key(), *_node->_value);
  }

  pointer operator->() const {
    assert(_node != nullptr && "dereferencing end() iterator");
    return std::make_pair(_node->get_key(), _node->_value);
  }

  self &operator++() {
    assert(_node != nullptr && "incrementing end() iterator");
    if (_node == _trie->_impl.root()) {
      auto it = _node->_children.begin();
      if (it != _node->_children.end()) {
        _node = it->second;
        _node = _node->leftmostfirst();
      } else {
        _node = nullptr;
      }
    } else if (!_node->_children.empty()) {
      auto it = _node->_children.begin();
      _node = it->second;
      _node = _node->leftmostfirst();
    } else {
      bool done = true;
      do {
        auto it = _node->_parent->_children.find(_node->_index);
        ++it;
        if (it != _node->_parent->_children.end()) {
          _node = it->second;
          if (_node->_value == nullptr) {
            _node = _node->leftmostfirst();
          }
          done = true;
        } else {
          _node = _node->_parent;
          if (_node->_parent == nullptr) {
            _node = nullptr;
            done = true;
          } else {
            done = false;
          }
        }
      } while (!done);
    }
    return *this;
  }

  self operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  self &operator--() {
    if (_node == _trie->_impl.root()) {
      assert(_node != nullptr && "decrementing begin() iterator");
      return *this;
    }

    if (_node == nullptr) {
      // start at rightmost node from root
      _node = _trie->_impl.root()->rightmostfirst();
      assert(_node->_value != nullptr);
      return *this;
    }

    while (_node->_parent != nullptr) {
      auto it = std::make_reverse_iterator(
          _node->_parent->_children.find(_node->_index));
      // check if we can go up and to the left
      if (it != _node->_parent->_children.rend()) {
        _node = it->second;
        _node = _node->rightmostfirst();
        break;
      } else {
        _node = _node->_parent;
        if (_node->_value != nullptr) {
          break;
        }
      }
    }
    return *this;
  }

  self operator--(int) {
    auto tmp = *this;
    --(*this);
    return tmp;
  }

  bool operator==(const self &other) const {
    return _trie == other._trie && _node == other._node;
  }

  bool operator!=(const self &other) const {
    return _trie != other._trie || _node != other._node;
  }
};

template <typename Key, typename Index, typename T, typename Allocator>
class trie_const_iterator {
private:
  using key_type = Key;
  using index_type = Index;
  using allocator_type = Allocator;
  using self = trie_const_iterator<Key, Index, T, Allocator>;

public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = std::pair<const key_type, const value_type *>;
  using reference = std::pair<const key_type, const value_type &>;

private:
  template <typename KeyT, typename TT, typename AllocatorT> friend class trie;

  const trie<Key, T, Allocator> *_trie;
  const trie_node<Key, Index, value_type> *_node;

public:
  trie_const_iterator() : _trie(nullptr), _node(nullptr) {}

  explicit trie_const_iterator(const trie<Key, T, Allocator> *trie,
                               const trie_node<Key, Index, value_type> *n)
      : _trie(trie), _node{n} {}

  trie_const_iterator(const trie_iterator<Key, Index, T, Allocator> iter)
      : _trie(iter._trie), _node{iter._node} {}

  trie_const_iterator(const trie_const_iterator &other)
      : _trie(other._trie), _node(other._node) {}

  trie_const_iterator &operator=(const trie_const_iterator &other) {
    if (this == &other) {
      return *this;
    }
    _trie = other._trie;
    _node = other._node;
    return *this;
  }

  reference operator*() const {
    assert(_node != nullptr && "dereferencing end() iterator");
    return std::make_pair(_node->get_key(), *_node->_value);
  }

  pointer operator->() const {
    assert(_node != nullptr && "dereferencing end() iterator");
    return std::make_pair(_node->get_key(), _node->_value);
  }

  self &operator++() {
    assert(_node != nullptr && "incrementing end() iterator");
    if (_node == _trie->_impl.root()) {
      auto it = _node->_children.begin();
      if (it != _node->_children.end()) {
        _node = it->second;
        _node = _node->leftmostfirst();
      } else {
        _node = nullptr;
      }
    } else if (!_node->_children.empty()) {
      auto it = _node->_children.begin();
      _node = it->second;
      _node = _node->leftmostfirst();
    } else {
      bool done = true;
      do {
        auto it = _node->_parent->_children.find(_node->_index);
        ++it;
        if (it != _node->_parent->_children.end()) {
          _node = it->second;
          if (_node->_value == nullptr) {
            _node = _node->leftmostfirst();
          }
          done = true;
        } else {
          _node = _node->_parent;
          if (_node->_parent == nullptr) {
            _node = nullptr;
            done = true;
          } else {
            done = false;
          }
        }
      } while (!done);
    }
    return *this;
  }

  self operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  self &operator--() {
    if (_node == _trie->_impl.root()) {
      assert(_node != nullptr && "decrementing begin() iterator");
      return *this;
    }

    if (_node == nullptr) {
      _node = _trie->_impl.root()->rightmostfirst();
      assert(_node->_value != nullptr);
      return *this;
    }

    while (_node->_parent != nullptr) {
      auto it = std::make_reverse_iterator(
          _node->_parent->_children.find(_node->_index));
      // check if we can go up and to the left
      if (it != _node->_parent->_children.rend()) {
        _node = it->second;
        _node = _node->rightmostfirst();
        break;
      } else {
        _node = _node->_parent;
        if (_node->_value != nullptr) {
          break;
        }
      }
    }
    return *this;
  }

  self operator--(int) {
    auto tmp = *this;
    --(*this);
    return tmp;
  }

  bool operator==(const self &other) const {
    return _trie == other._trie && _node == other._node;
  }

  bool operator!=(const self &other) const {
    return _trie != other._trie || _node != other._node;
  }
};

// MARK: - Implementation

template <typename Key, typename T, typename Allocator>
class trie_impl : public Allocator {
private:
  using key_type = Key;
  using index_type = typename Key::value_type;
  using mapped_type = T;
  using tree_node = trie_node<Key, index_type, T>;

private:
  tree_node *_root;
  std::size_t _size;

public:
  explicit trie_impl(const Allocator &a = Allocator())
      : Allocator(a), _root(new tree_node()), _size(0) {}

  trie_impl(const trie_impl &other)
      : Allocator(other), _root(new tree_node()), _size(other._size) {
    _root = other._root->clone(*static_cast<Allocator *>(this));
  }

  trie_impl(trie_impl &&other)
      : Allocator(other), _root(other._root), _size(other._size) {
    other._root = nullptr;
    other._size = 0;
  }

  trie_impl &operator=(const trie_impl &other) {
    if (this != &other) {
      *static_cast<Allocator *>(this) = other;
      _root = other._root->clone(*static_cast<Allocator *>(this));
      _size = other._size;
    }
    return *this;
  }

  trie_impl &operator=(trie_impl &&other) {
    if (this != &other) {
      *static_cast<Allocator *>(this) =
          std::move(static_cast<Allocator>(other));
      if (_root != nullptr) {
        clear();
        delete _root;
      }
      _root = other._root;
      other._root = nullptr;
      _size = other._size;
      other._size = 0;
    }
    return *this;
  }

  ~trie_impl() {
    clear();
    delete _root;
    _root = nullptr;
  }

  tree_node *root() { return _root; }

  const tree_node *root() const { return _root; }

  tree_node *find(const key_type &key) {
    auto current_node = _root;
    for (const auto character : key) {
      auto it = current_node->_children.find(character);
      if (it == current_node->_children.end()) {
        return nullptr;
      }
      current_node = it->second;
    }
    if (current_node->_value == nullptr) {
      return nullptr;
    }
    return current_node;
  }

  const tree_node *find(const key_type &key) const {
    auto current_node = _root;
    for (const auto index : key) {
      auto it = current_node->_children.find(index);
      if (it == current_node->_children.end()) {
        return nullptr;
      }
      current_node = it->second;
    }
    if (current_node->_value == nullptr) {
      return nullptr;
    }
    return current_node;
  }

  std::pair<tree_node *, bool> insert(const key_type &key,
                                      const mapped_type &value) {
    auto current_node = _root;
    auto parent_node = current_node;
    for (const auto character : key) {
      parent_node = current_node;
      current_node = parent_node->_children[character];
      if (current_node == nullptr) {
        current_node = parent_node->_children[character] =
            new tree_node(parent_node, character);
      }
    }
    if (current_node->_value != nullptr) {
      return std::make_pair(current_node, false);
    }
    current_node->set_value(*static_cast<Allocator *>(this), key, value);
    _size += 1;
    return std::make_pair(current_node, true);
  }

  void erase(tree_node *node) {
    _size -= 1;
    if (!node->is_leaf()) {
      node->remove_value(*static_cast<Allocator *>(this));
      return;
    }
    if (node->_parent == nullptr) {
      node->_value = nullptr;
      return;
    }
    auto child = node;
    auto parent = node->_parent;
    while (parent->_parent != nullptr && parent->_children.size() == 1 &&
           parent->_value == nullptr) {
      child = parent;
      parent = parent->_parent;
    }
    child->clean_recursively(*static_cast<Allocator *>(this));
    parent->_children.erase(child->_index);
  }

  void clear() {
    if (_root != nullptr) {
      _root->clean_recursively(*static_cast<Allocator *>(this));
      _size = 0;
    }
  }

  std::size_t size() const { return _size; }

  bool empty() const { return _size == 0; }
};

template <typename Key, typename Index, typename Value> struct trie_node {
  using value_type = Value;
  using index_type = Index;
  using key_type = Key;
  using children_type = std::map<index_type, trie_node *>;

  explicit trie_node(trie_node *parent = nullptr,
                     const index_type key = index_type(),
                     value_type *value = nullptr)
      : _parent{parent}, _index{key}, _value{value} {}

  trie_node *_parent;
  index_type _index;
  value_type *_value;
  children_type _children;

  trie_node(const trie_node &) = delete;
  trie_node(trie_node &&) = delete;
  trie_node &operator=(const trie_node &) = delete;
  trie_node &operator=(trie_node &&) = delete;
  ~trie_node() = default;
  
  
  key_type get_key() const;

  template <typename AllocatorT, typename... Args>
  void set_value(AllocatorT alloc, Args &&...args) {
    if (_value != nullptr) {
      remove_value(alloc);
    }
    _value = alloc.allocate(1);
    std::allocator_traits<AllocatorT>::construct(alloc, _value,
                                                 std::forward<Args>(args)...);
  }

  template <typename AllocatorT> void remove_value(AllocatorT alloc) {
    std::allocator_traits<AllocatorT>::destroy(alloc, _value);
    alloc.deallocate(_value, 1);
    _value = nullptr;
  }

  template <typename AllocatorT> void clean_recursively(AllocatorT alloc) {
    for (auto &[index, node] : _children) {
      node->clean_recursively(alloc);
      delete node;
      node = nullptr;
    }
    _children.clear();
    if (_value != nullptr) {
      remove_value(alloc);
    }
  }

  template <typename AllocatorT>
  trie_node *clone(AllocatorT alloc, trie_node *parent = nullptr) {
    trie_node *result = new trie_node(parent, _index);
    if (_value != nullptr) {
      result->set_value(alloc, *_value);
    }
    for (const auto &[index, node] : _children) {
      result->_children[index] = node->clone(alloc, result);
    }
    return result;
  }

  trie_node *leftmostfirst() {
    return const_cast<trie_node *>(
        const_cast<const trie_node *>(this)->leftmostfirst());
  }

  const trie_node *leftmostfirst() const {
    auto result = this;
    while (result->_value == nullptr) {
      auto it = result->_children.begin();
      if (it == result->_children.end()) {
        break;
      }
      result = it->second;
    }
    return result;
  }

  trie_node *rightmostfirst() {
    return const_cast<trie_node *>(
        const_cast<const trie_node *>(this)->rightmostfirst());
  }

  const trie_node *rightmostfirst() const {
    auto result = this;
    while (!result->_children.empty()) {
      auto it = result->_children.rbegin();
      if (it == result->_children.rend()) {
        break;
      }
      result = it->second;
    }
    return result;
  }

  key_type key() const {
    key_type result;
    auto current_node = this;
    while (current_node->_parent != nullptr) {
      result.push_back(current_node->_index);
      current_node = current_node->_parent;
    }
    std::reverse(result.begin(), result.end());
    return std::move(result);
  }

  bool is_leaf() const { return _children.empty(); }
};

// MARK: - Interface

template <typename Key, typename T,
          typename Allocator = std::allocator<T>>
class trie {
public:
  using key_type = Key;
  using value_type = T;
  using index_type = typename key_type::value_type;
  using size_type = std::size_t;
  using allocator_type = Allocator;
  using reference = std::pair<const key_type, value_type&>;
  using const_reference = std::pair<const key_type, const value_type&>;
  using pointer = std::pair<const key_type, const value_type*>;
  using const_pointer = std::pair<const key_type, const value_type*>;
  using iterator = trie_iterator<key_type, index_type, value_type, allocator_type>;
  using const_iterator = trie_const_iterator<key_type, index_type, value_type, allocator_type>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:
  using self = trie<key_type, value_type, allocator_type>;

private:
  using impl_type = trie_impl<key_type, value_type, allocator_type>;

private:
  impl_type _impl;

public:
  // MARK: Constructors, assignment and destructor
  explicit trie(const allocator_type &a = allocator_type()) : _impl(a) {}

  trie(const trie &) = default;
  trie(trie &&) = default;
  trie &operator=(const trie &) = default;
  trie &operator=(trie &&) = default;
  ~trie() = default;

  trie(std::initializer_list<value_type> il,
       const allocator_type &a = allocator_type())
      : _impl(a) {
    for (const auto &[key, value] : il) {
      _impl.insert(key, value);
    }
  }

  template <typename FwdIterT>
  trie(FwdIterT first, FwdIterT last,
       const allocator_type &a = allocator_type())
      : _impl(a) {
    for (auto it = first; it != last; ++it) {
      _impl.insert(it->first, it->second);
    }
  }

  // MARK: Selectors
  const_iterator find(const key_type &key) const {
    auto node = _impl.find(key);
    return const_iterator(this, node);
  }

  size_type size() const { return _impl.size(); }

  size_type max_size() const { return std::numeric_limits<size_type>::max(); }

  bool empty() const { return _impl.empty(); }

  reference at(const key_type &key) {
    return const_cast<const_reference>(const_cast<self *>(this)->at(key));
  }

  const_reference at(const key_type &key) const {
    auto node = _impl.find(key);
    if (node == nullptr) {
      llvm::report_fatal_error("out_of_range trie::at");
    }

    return node->_value->second;
  }

  // MARK: Mutators
  iterator find(const key_type &key) {
    auto node = _impl.find(key);
    return iterator(this, node);
  }

  std::pair<iterator, bool> insert(const key_type &key,
                                   const value_type &value) {
    auto [node, inserted] = _impl.insert(key, value);
    return std::make_pair(iterator(this, node), inserted);
  }

  std::pair<iterator, bool> insert(const value_type &value) {
    return insert(value.first, value.second);
  }

  iterator insert(const_iterator hint, const value_type &value) {
    return insert(value).first;
  }

  reference operator[](const key_type &key) {
    auto node = _impl.find(key);
    if (node == nullptr) {
      node = _impl.insert(key, value_type()).first;
    }

    return node->_value->second;
  }

  void erase(const key_type &key) {
    auto node = _impl.find(key);
    if (node != nullptr) {
      _impl.erase(node);
    }
  }

  iterator erase(iterator pos) {
    auto it = pos;
    ++it;
    _impl.erase(pos._node);
    return it;
  }

  void clear() { _impl.clear(); }

  // MARK: Iterators
  iterator begin() {
    if (empty()) {
      return end();
    } else {
      return iterator(this, _impl.root()->leftmostfirst());
    }
  }

  const_iterator begin() const {
    if (empty()) {
      return end();
    } else {
      return const_iterator(this, _impl.root()->leftmostfirst());
    }
  }

  const_iterator cbegin() const { return begin(); }

  iterator end() { return iterator(this, nullptr); }

  const_iterator end() const { return const_iterator(this, nullptr); }

  const_iterator cend() const { return end(); }

  reverse_iterator rbegin() { return std::make_reverse_iterator(end()); }

  const_reverse_iterator rbegin() const {
    return std::make_reverse_iterator(end());
  }

  const_reverse_iterator crbegin() const { return rbegin(); }

  reverse_iterator rend() { return std::make_reverse_iterator(begin()); }

  const_reverse_iterator rend() const {
    return std::make_reverse_iterator(begin());
  }

  const_reverse_iterator crend() const { return rend(); }

  // MARK: Allocator
  allocator_type get_allocator() const {
    return static_cast<allocator_type>(_impl);
  }

  bool operator==(const self &other) const {
    return std::equal(begin(), end(), other.begin(), other.end());
  }

  bool operator!=(const self &other) const { return !(*this == other); }

  bool operator<(const self &other) const {
    auto cmp = [](const self::value_type &lhs, const self::value_type &rhs) {
      return lhs.first < rhs.first;
    };
    return std::lexicographical_compare(begin(), end(), other.begin(),
                                        other.end(), cmp);
  }

  bool operator<=(const self &rhs) const { return !(*this > rhs); }

  bool operator>(const self &other) {
    auto cmp = [](const self::value_type &lhs, const self::value_type &rhs) {
      return lhs.first > rhs.first;
    };
    return std::lexicographical_compare(begin(), end(), other.begin(),
                                        other.end(), cmp);
  }

  bool operator>=(const self &rhs) { return !(*this < rhs); }
};
