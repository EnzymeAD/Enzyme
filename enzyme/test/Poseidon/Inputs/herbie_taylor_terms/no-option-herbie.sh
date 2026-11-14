#!/bin/sh
# A Herbie that predates --num-taylor-terms, as far as the option goes.
case " $* " in
*" --help "*) echo "  --num-enodes <num>" ;;
*" --num-taylor-terms "*)
  echo "herbie report: unknown switch: --num-taylor-terms" >&2
  exit 1
  ;;
esac
