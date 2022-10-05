import json
import datetime
import subprocess
import requests
import argparse
import platform

def get_git_revision_short_hash():
    try:
        output = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.STDOUT).decode('ascii').strip()
    except:
        return None

def extract_results(json, llvm, mode):
    result = []
    githash = get_git_revision_short_hash()
    time = datetime.datetime.now().replace(microsecond=0).isoformat()
    cpu = platform.processor()
    
    for test_suite in json:
        series = test_suite["name"]
        for tool in test_suite["tools"]:
            if "enzyme" in tool["name"].lower():
                value = tool["runtime"]
                res = {
                    "mode": mode,
                    "llvm_version": llvm,
                    "testsuite": series,
                    "commit": githash,
                    "name": tool,
                    "elapsed": value,
                    "timestamp": time,
                    "platform": cpu
                }
                result.append(res)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('file', type=argparse.FileType('r'), help="JSON file containing the benchmark results")
parser.add_argument('-t', '--token', type=str, required=False, help="Token to authorize at graphing backend")
parser.add_argument('-u', '--url', type=str, required=True, help="URL of the graphing backend")

parser.add_argument('--llvm', type=str, required=True, help="LLVM Version")
parser.add_argument('--mode', type=str, required=True, help="Derivative Mode")


args = parser.parse_args()

json = json.load(args.file)
results = extract_results(json, args.llvm, args.mode)

if args.token:
    requests.post(args.url, json=results, headers={"TOKEN": args.token})
else:
    requests.post(args.url, json=results)