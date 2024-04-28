#!/usr/bin/env sh

set -e

INPUT_DIR=$(dirname $1)/..
OUTPUT_FILE=$2

echo $INPUT_FILE
echo $OUTPUT_FILE
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo > "$OUTPUT_FILE"
echo "const char* include_headers[][2] = {" >> "$OUTPUT_FILE"
for FILE in $(find -L $INPUT_DIR -type f); do
    echo $FILE
    INTERNAL_FILENAME=$(echo $FILE | sed "s|$INPUT_DIR|\/enzymeroot\/|")
    echo $INTERNAL_FILENAME
    echo '{"'"$INTERNAL_FILENAME"'",' >> "$OUTPUT_FILE"
    echo 'R"(' >> "$OUTPUT_FILE"
    cat $FILE >> "$OUTPUT_FILE"
    echo ')"' >> "$OUTPUT_FILE"
    echo '},' >> "$OUTPUT_FILE"
done
echo '};' >> "$OUTPUT_FILE"
