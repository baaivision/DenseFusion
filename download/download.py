import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import requests


def run(inp):
    x, args = inp
    jx = json.loads(x)

    out_path = f"{args.output_path}/{jx['image_id']}.png"
    if osp.exists(out_path):
        jx["image"] = out_path
        return json.dumps(jx)

    try:
        url = jx["url"]
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            jx["image"] = out_path
            return json.dumps(jx)
        else:
            return None
    except Exception as ex:
        print(ex)
        return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--json-file', type=str, required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--json-out-file', type=str, required=True)
    parser.add_argument('--num-processes', default=2, type=int)

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    xs = [[x.strip(), args] for x in open(args.json_file).readlines()]

    p = mp.Pool(args.num_processes)
    r = p.map(run, xs)
    with open(args.json_out_file, 'w') as f:
        for rr in r:
            if rr is None:
                continue
            print(rr, file=f)


if __name__ == "__main__":
    main()
