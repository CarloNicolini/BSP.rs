"""Paired full solves: require identical results, histories, and assignments.

Run from BSP.rs after preserving the old binary, then building the changed one.
Elapsed time excludes process startup, formula parsing, and topology creation;
solver wall_seconds includes native state construction and final verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile


def result(command):
    process = subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=120)
    if process.returncode not in (0,1):
        raise RuntimeError(process.stderr)
    return json.loads(process.stdout)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--after",type=Path,default=Path("target/release/bsp-rust"))
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.repeats<1: parser.error("repeats must be positive")
    rows=[]
    with tempfile.TemporaryDirectory(prefix="bsp-indexing-") as tmp:
        for k,alpha,n,seed in [(3,2.5,400,3),(3,4.15,400,3),(3,4.15,400,1),(4,9.5,400,3),(3,4.2,1000,1)]:
            formula=Path(tmp)/"formula.cnf"
            subprocess.run([str(args.before),"--generate",str(k),str(alpha),str(n),"--seed",str(seed),"--write-formula",str(formula)],check=True)
            times={"before":[],"after":[]}
            outputs={}
            for trial in range(args.repeats+1):
                for key in (["before","after"] if trial%2 else ["after","before"]):
                    binary=args.before if key=="before" else args.after
                    output=result([binary,"--seed",seed,"--max-flips",100000,formula])
                    elapsed=output.pop("wall_seconds")
                    # Execution metadata is not part of the SAT trajectory.
                    output.pop("execution",None)
                    outputs[key]=output
                    if trial: times[key].append(elapsed)
                if outputs["before"]!=outputs["after"]:
                    raise RuntimeError(f"solver trajectory changed for {(k,alpha,n,seed)}")
            medians={key:statistics.median(value) for key,value in times.items()}
            r=outputs["after"]
            rows.append({"k":k,"alpha":alpha,"n":n,"seed":seed,"status":r["status"],"sp_sweeps":r["sp_sweeps"],
                "decimations":r["decimations"],"releases":r["releases"],"seconds":medians,"samples":times,
                "speedup":medians["before"]/medians["after"],"identical_trajectory":True})
    files=[args.before,args.after,Path("src/solver.rs"),Path("src/bitset.rs"),Path("src/decision_heap.rs"),Path("Cargo.lock")]
    report={"platform":platform.platform(),"rustc":subprocess.check_output(["rustc","--version"],text=True).strip(),
        "build":"Release opt-level=3, ThinLTO, one codegen unit; generic target for both binaries",
        "sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},"rows":rows}
    text=json.dumps(report,indent=2)+"\n"
    print(text,end="")
    if args.output: args.output.write_text(text)


if __name__=="__main__": main()
