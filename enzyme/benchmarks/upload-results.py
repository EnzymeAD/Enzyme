import json
import datetime
import subprocess
import requests
import argparse
import os

def get_git_revision_short_hash():
    try:
        output = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.STDOUT).decode('ascii').strip()
    except:
        return None

def extract_results(json):
    result = []
    githash = get_git_revision_short_hash()
    time = datetime.datetime.now().isoformat()
    metadata = {}
    if githash is not None:
        metadata["githash"] = githash
    
    for test_suite in json:
        series = test_suite["name"]
        for tool in test_suite["tools"]:
            if "enzyme" in tool["name"].lower():
                value = tool["runtime"]
                res = {"series": series, "value": value, "metadata": metadata, "timestamp": time}
                result.append(res)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('file', type=argparse.FileType('r'), help="JSON file containing the benchmark results")
parser.add_argument('-t', '--token', type=str, required=False, help="Token to authorize at graphing backend")
parser.add_argument('-u', '--url', type=str, required=True, help="URL of the graphing backend")

args = parser.parse_args()

json = json.load(args.file)
results = extract_results(json)

if args.token:
    requests.post(args.url, json=results, headers={"TOKEN": args.token})
else:
    requests.post(args.url, json=results)