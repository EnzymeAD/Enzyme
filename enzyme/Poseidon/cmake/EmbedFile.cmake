# Writes the bytes of IN to OUT as the C array NAME.
file(READ ${IN} hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(WRITE ${OUT} "static const unsigned char ${NAME}[] = {${bytes}};\n")
