"""Build an isolated C++ benchmark extension with phase timers and output controls.

The source checkout and installed package remain unchanged. At runtime set
BSP_BENCH_NUMSOL=1 or 10 and BSP_BENCH_WHITEN=0 or 1. Defaults preserve 10/1.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    args = parser.parse_args()
    args.build_root.mkdir(parents=True, exist_ok=False)
    source = args.build_root / "source"
    source.mkdir()
    for name in ["CMakeLists.txt", "pyproject.toml"]:
        shutil.copy2(args.source / name, source / name)
    for name in ["src", "include", "python/bindings"]:
        shutil.copytree(args.source / name, source / name)
    patches = []

    def edit(name, replacements):
        path = source / name
        before = path.read_text()
        after = before
        for old, new in replacements:
            assert after.count(old) == 1, (name, old, after.count(old))
            after = after.replace(old, new)
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=f"a/{name}", tofile=f"b/{name}"))

    edit("src/run_algorithm.cpp", [
        ("#include <bsp/Graph.hpp>", "#include <bsp/Graph.hpp>\n#include <chrono>"),
        ("    V.resize(G.N());", "    const auto bench_bsp_start=std::chrono::steady_clock::now();\n    V.resize(G.N());"),
        ("    bool solved=G.WalkSAT();", '    bsp::console_fprintf(stderr,"BSP_BENCH bsp_seconds %.17g\\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-bench_bsp_start).count());\n    bool solved=G.WalkSAT();'),
    ])
    edit("src/walksat.cpp", [
        ("int WalkSat(std::vector <std::vector<bool> > & sol,int argc, char * argv[])\n{", "int WalkSat(std::vector <std::vector<bool> > & sol,int argc, char * argv[])\n{\n    const auto bench_walk_start=std::chrono::steady_clock::now();\n    double bench_walk_first=-1.;"),
        ("        update_and_print_statistics_end_try(sol);", "        update_and_print_statistics_end_try(sol);\n        if(numsuccesstry==1 && bench_walk_first<0.) bench_walk_first=std::chrono::duration<double>(std::chrono::steady_clock::now()-bench_walk_start).count();"),
        ("    expertime = elapsed_seconds();", '    bsp::console_fprintf(stderr,"BSP_BENCH walksat_first_seconds %.17g\\nBSP_BENCH walksat_all_seconds %.17g\\n",bench_walk_first,std::chrono::duration<double>(std::chrono::steady_clock::now()-bench_walk_start).count());\n    expertime = elapsed_seconds();'),
        ('std::vector<std::string> args={"walksat","-solcnf","-cutoff","100000000","-best","-seed","1234567","-numsol","10"};', 'const char *bench_target_env=std::getenv("BSP_BENCH_NUMSOL");\n    std::string bench_target=bench_target_env ? bench_target_env : "10";\n    if(bench_target!="1" && bench_target!="10") throw std::runtime_error("BSP_BENCH_NUMSOL must be 1 or 10");\n    std::vector<std::string> args={"walksat","-solcnf","-cutoff","100000000","-best","-seed","1234567","-numsol",bench_target};'),
    ])
    # chrono/cstdlib may be provided transitively; explicit includes make the
    # instrumentation independent of unrelated header contents.
    walk = source / "src/walksat.cpp"
    walk.write_text("#include <chrono>\n#include <cstdlib>\n" + walk.read_text())
    edit("src/Graph.cpp", [
        ("    } else WalkSat(sol,argc,argv);/*call WalkSAT*/\n    vector<long int> mysol;", "    } else WalkSat(sol,argc,argv);/*call WalkSAT*/\n    const auto bench_post_start=std::chrono::steady_clock::now();\n    double bench_whitening_seconds=0.;\n    vector<long int> mysol;"),
        ("        Whitening_Solution();/*whitening procedure on a complete solution*/", '        const char *bench_whiten=std::getenv("BSP_BENCH_WHITEN");\n        if(!bench_whiten || std::string(bench_whiten)!="0") {\n            const auto bench_white_start=std::chrono::steady_clock::now();\n            Whitening_Solution();/*whitening procedure on a complete solution*/\n            bench_whitening_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-bench_white_start).count();\n        }'),
        ("    return flag;\n}\n\n/*Frozen and unfrozen variables and whitening procedure*/", '    bsp::console_fprintf(stderr,"BSP_BENCH postprocess_seconds %.17g\\nBSP_BENCH whitening_seconds %.17g\\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-bench_post_start).count(),bench_whitening_seconds);\n    return flag;\n}\n\n/*Frozen and unfrozen variables and whitening procedure*/'),
    ])
    graph = source / "src/Graph.cpp"
    graph.write_text("#include <cstdlib>\n" + graph.read_text())
    # Recreate the complete patch, including the two added include blocks.
    patches = []
    for name in ["src/run_algorithm.cpp", "src/walksat.cpp", "src/Graph.cpp"]:
        patches.extend(difflib.unified_diff((args.source/name).read_text().splitlines(True), (source/name).read_text().splitlines(True), fromfile=f"a/{name}", tofile=f"b/{name}"))
    (args.build_root / "benchmark.patch").write_text("".join(patches))
    build = args.build_root / "build"
    configure = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DBSP_BUILD_PYTHON=ON", "-DBSP_BUILD_CLI=OFF", "-DBSP_BUILD_TESTS=OFF", "-DBSP_NATIVE_ARCH=OFF", "-DBSP_ENABLE_OPENMP=OFF", "-DBSP_IDE_LINK=OFF", f"-DPython_EXECUTABLE={sys.executable}"]
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    binary, = build.glob("_native*.so")
    metadata = dict(source=str(args.source), configure=configure, binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), patch="benchmark.patch")
    (args.build_root / "build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
