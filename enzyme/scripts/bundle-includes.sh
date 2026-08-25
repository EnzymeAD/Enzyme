#!/bin/bash

set -e

INPUT_DIR=$(dirname "${1}")/..
OUTPUT_FILE=${2}

echo "${INPUT_FILE}"
echo "${OUTPUT_FILE}"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo >"${OUTPUT_FILE}"
echo "const char* include_headers[][2] = {" >>"${OUTPUT_FILE}"

find -L "${INPUT_DIR}" -type f | while read -r FILE; do
  echo "${FILE}"
  INTERNAL_FILENAME="${FILE/${INPUT_DIR}/\/enzymeroot\/}"
  echo "${INTERNAL_FILENAME}"
  {
    echo '{"'"${INTERNAL_FILENAME}"'",'
    echo 'R"('
    cat "${FILE}"
    echo ')"'
    echo '},'
  } >>"${OUTPUT_FILE}"
done

echo '};' >>"${OUTPUT_FILE}"
