#pragma once

bool any_tests_failed = false; 

#define EXPECT(boolean) \
if (!(boolean)) { \
  std::cout << "Test failure on " << __FILE__ << ":" << __LINE__ << ", EXPECT(" << (#boolean) << ")" << std::endl; \
  any_tests_failed = true; \
}