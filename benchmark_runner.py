import time
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, CXGate

# ==========================================
# IMPORTS FROM YOUR EXISTING FILES
# ==========================================
# 1. The Dense Matrix (Original Paper) Implementation
from qiskit_static_analyzer import QuantumStaticAnalyzer

# 2. The SparsePauliOp Implementation
from new_implementation_sparsepauliop import AbstractState

# ==========================================
# CIRCUIT & DOMAIN HELPERS
# ==========================================
def build_chain_ghz(n):
    """Nearest-neighbor GHZ circuit to match sliding-window domains."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def get_sliding_domain(n, k):
    """
    Generates a domain like[0,1,2], [2,3,4] for k=3.
    Steps by k-1 to ensure exactly 1 qubit overlaps between subsets,
    maintaining the 'Connected' requirement of the paper.
    """
    domain =[]
    for i in range(0, n - k + 1, k - 1):
        domain.append(tuple(range(i, i + k)))
        
    # Catch remaining qubits at the end
    if domain and domain[-1][-1] < n - 1:
        domain.append(tuple(range(n - k, n)))
        
    return list(set(domain))

# ==========================================
# BENCHMARK RUNNER
# ==========================================
def run_comparison_benchmark():
    N = 300
    k_values = [2, 3, 4, 5, 6, 7]  # Testing subset sizes from 2 to 6
    
    # Pre-build circuit and matrix gates for the runner
    qc = build_chain_ghz(N)
    h_matrix = HGate().to_matrix()
    cx_matrix = CXGate().to_matrix()
    
    times_dense =[]
    times_sparse =[]
    
    print(f"--- Running Benchmark for N={N} Qubits ---")
    print(f"{'Subset Size (k)':<15} | {'Dense Time':<15} | {'SparsePauli Time':<15}")
    print("-" * 55)
    
    for k in k_values:
        domain = get_sliding_domain(N, k)
        
        # ----------------------------------------------------
        # 1. Run Dense Method (from qiskit_static_analyzer.py)
        # ----------------------------------------------------
        start = time.perf_counter()
        analyzer_dense = QuantumStaticAnalyzer(domain)
        
        # This implementation requires manual gate application
        analyzer_dense.apply_gate(h_matrix, [0])
        for i in range(N - 1):
            analyzer_dense.apply_gate(cx_matrix, [i, i + 1])
            
        # Assertion Checking
        is_verified_dense = True
        for subset in analyzer_dense.domain:
            k_sub = len(subset)
            size = 2**k_sub
            proj_mat = np.zeros((size, size), dtype=complex)
            proj_mat[0, 0] = 1.0
            proj_mat[size-1, size-1] = 1.0
            if not analyzer_dense.check_assertion(proj_mat, list(subset)):
                is_verified_dense = False
                break
                
        t_dense = time.perf_counter() - start
        times_dense.append(t_dense)
        
        # ----------------------------------------------------
        # 2. Run SparsePauli Method (from new_implementation_sparsepauliop.py)
        # ----------------------------------------------------
        start = time.perf_counter()
        analyzer_sparse = AbstractState(qc, domain)
        
        # This implementation has an automated circuit processor
        analyzer_sparse.process_circuit(qc)
        
        # Assertion Checking
        from qiskit.quantum_info import SparsePauliOp
        is_verified_sparse = True
        for subset in analyzer_sparse.domain:
            k_sub = len(subset)
            size = 2**k_sub
            proj_mat = np.zeros((size, size), dtype=complex)
            proj_mat[0, 0] = 1.0
            proj_mat[size-1, size-1] = 1.0
            pauli_projector = SparsePauliOp.from_operator(proj_mat)
            
            local_dm = analyzer_sparse.state[subset]
            exp_val = np.real(local_dm.expectation_value(pauli_projector))
            if not np.isclose(exp_val, 1.0, atol=1e-5):
                is_verified_sparse = False
                break
                
        t_sparse = time.perf_counter() - start
        times_sparse.append(t_sparse)
        
        print(f"{k:<15} | {t_dense:<11.4f} sec | {t_sparse:<11.4f} sec | Dense Pass: {is_verified_dense} | Sparse Pass: {is_verified_sparse}")
        
    # Trigger the plotting function
    plot_results(k_values, times_dense, times_sparse, N)


def plot_results(k_values, times_dense, times_sparse, N):
    """Generates a logarithmic plot with growth multipliers annotated."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data on a Logarithmic Y-axis
    ax.semilogy(k_values, times_dense, marker='o', linewidth=2, label="Paper Implementation (Dense Matrices)", color='red')
    ax.semilogy(k_values, times_sparse, marker='s', linewidth=2, label="SparsePauliOp Implementation", color='blue')
    
    # Add slope annotations (Growth Multipliers)
    for i in range(1, len(k_values)):
        mid_x = (k_values[i-1] + k_values[i]) / 2
        
        # Calculate growth multiplier: Time(k) / Time(k-1)
        mult_dense = times_dense[i] / times_dense[i-1]
        mult_sparse = times_sparse[i] / times_sparse[i-1]
        
        # Determine geometric midpoint for Y-axis placing on log scale
        mid_y_dense = np.sqrt(times_dense[i-1] * times_dense[i])
        mid_y_sparse = np.sqrt(times_sparse[i-1] * times_sparse[i])
        
        ax.annotate(f"{mult_dense:.1f}x", xy=(mid_x, mid_y_dense), 
                    xytext=(0, 10), textcoords='offset points', color='red', ha='center', fontsize=11, fontweight='bold')
                    
        ax.annotate(f"{mult_sparse:.1f}x", xy=(mid_x, mid_y_sparse), 
                    xytext=(0, -15), textcoords='offset points', color='blue', ha='center', fontsize=11, fontweight='bold')

    ax.set_title(f"Static Analysis Scaling over {N} Qubits: Dense vs SparsePauliOp", fontsize=14, pad=15)
    ax.set_xlabel("Abstract Subset Size (k qubits per domain)", fontsize=12)
    ax.set_ylabel("Total Execution Time (seconds) - Log Scale", fontsize=12)
    ax.set_xticks(k_values)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    filename = f"timing_comparison_plot_{N}.png"
    plt.savefig(filename, dpi=300)
    print(f"\nPlot saved successfully to '{filename}'")
    plt.show()

if __name__ == "__main__":
    run_comparison_benchmark()