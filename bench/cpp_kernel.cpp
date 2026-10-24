// Baseline harness for the zero-temperature SP loop in src/Graph.cpp.
// Uses the existing Graph/Vertex storage and product update. Omits logging,
// convergence exits, unit propagation, and complexity estimation for both kernels.
// Build with the same Release/native flags as bsp_core. No solver code is changed.
#include <bsp/Graph.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <stdexcept>

int main(int argc, char **argv) {
    if (argc != 4) throw std::runtime_error("cpp_kernel formula.cnf sweeps damping");
    bsp::Formula f;
    std::ifstream input(argv[1]);
    std::string line;
    unsigned declared = 0;
    std::vector<int> clause;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == 'c') continue;
        std::istringstream row(line);
        if (line[0] == 'p') { std::string p, cnf; row >> p >> cnf >> f.n >> declared; }
        else { int lit; while (row >> lit) { if(lit) clause.push_back(lit); else { f.clauses.push_back(clause); clause.clear(); } } }
    }
    if (f.clauses.size() != declared || f.n == 0) throw std::runtime_error("invalid benchmark formula");
    g_fixed_seed = 1;
    Graph graph(1, argv);
    graph.load_formula(f);
    std::vector<Vertex> vertices(f.n);
    for (unsigned v = 0; v < f.n; ++v) vertices[v]._vertex = v+1;
    graph.get_V(vertices);
    graph.split_and_collect_information();
    size_t edge = 0;
    for (auto &cl : graph._cl) for (unsigned i = 0; i < cl._size_cl_init; ++i, ++edge) {
        const double eta = 0.05+0.25*double((edge*37)%997)/997.;
        *cl.v_survey_cl_to_i[i] = cl.update[i] = eta;
        cl.div_s[i] = 1./(1.-eta);
    }
    graph.update_products();
    const unsigned sweeps = std::stoul(argv[2]);
    const double damping = std::stod(argv[3]);
    double residual = 0.;
    auto start = std::chrono::steady_clock::now();
    for (unsigned t = 0; t < sweeps; ++t) {
        residual = 0.;
        for (auto &cl : graph._cl) {
            const unsigned k = cl._size_cl_init;
            for (unsigned i = 0; i < k; ++i) {
                double numerator = 1., denominator = 1.;
                for (unsigned j = 0; j < k; ++j) if(i != j) {
                    const Vertex *v = cl.v_V[j];
                    const double ps = (cl.v_lit[j] ? v->prod_V_plus : v->prod_V_minus)*cl.div_s[j];
                    const double pu = cl.v_lit[j] ? v->prod_V_minus : v->prod_V_plus;
                    numerator *= (1.-pu)*ps;
                    denominator *= ps+pu-ps*pu;
                }
                double raw = numerator/denominator;
                if (raw < ZERO) raw = 0.;
                residual = std::max(residual, std::abs(raw-cl.update[i]));
                cl.update[i] = raw+(cl.update[i]-raw)*damping;
            }
            for (unsigned i = 0; i < k; ++i) {
                *cl.v_survey_cl_to_i[i] = cl.update[i];
                cl.div_s[i] = 1./(1.-cl.update[i]);
            }
        }
        graph.update_products();
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double checksum = 0.;
    for (auto &cl : graph._cl) for(double value : cl.update) checksum += value;
    std::cout << std::setprecision(17) << "{\"seconds\":" << seconds
        << ",\"ns_per_edge_sweep\":" << seconds*1e9/(sweeps*double(edge))
        << ",\"residual\":" << residual << ",\"checksum\":" << checksum << ",\"messages\":[";
    bool first = true;
    for (auto &cl : graph._cl) for(double value : cl.update) {
        if (!first) std::cout << ',';
        first = false; std::cout << value;
    }
    std::cout << "]}\n";
}
