import sys

for i in range(1,len(sys.argv)):
  content = open('test.template').read()
  content = content.replace("@NAME@", sys.argv[i])
  open('./testfiles/'+sys.argv[i]+".test", 'w+').write(content)
