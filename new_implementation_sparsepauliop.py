import time
import numpy as np
from scipy.linalg import null_space

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Operator, partial_trace

# ==========================================
# 1. DYNAMIC DOMAIN EXTRACTOR
# ==========================================
def generate_domain_from_circuit(circuit: QuantumCircuit) -> list[tuple[int, ...]]:
    """
    Automatically extracts the necessary domain subsets by tracking all 
    multi-qubit interactions in the circuit (Data-Flow Heuristic).
    """
    domain_set = set()
    for instruction in circuit.data:
        # Ignore barriers and measurements
        if instruction.operation.name in['barrier', 'measure', 'reset', 'delay']:
             continue
        if len(instruction.qubits) >= 2:
            qubit_indices = tuple(sorted(circuit.find_bit(q).index for q in instruction.qubits))
            domain_set.add(qubit_indices)
            
    # Fallback for circuits with purely single-qubit gates
    if not domain_set:
        domain_set = {(circuit.find_bit(q).index,) for q in circuit.qubits}
        
    return list(domain_set)

# ==========================================
# 2. THE ABSTRACT STATE CLASS
# ==========================================
class AbstractState:
    def __init__(self, qc, domain):
        self.n = qc.num_qubits
        self.domain = domain
        self.state = {}

        # Initialize each subset dynamically based on its size
        for s in self.domain:
            k = len(s)
            size = 2**k
            init_matrix = np.zeros((size, size), dtype=complex)
            init_matrix[0, 0] = 1.0  # Initial state is always |0...0><0...0|
            
            self.state[s] = DensityMatrix(init_matrix)

    def _get_support(self, rho_dm):
        """Converts a fuzzy DensityMatrix into a crisp Projector by taking its support."""
        evals, evecs = np.linalg.eigh(rho_dm.data)
        mask = evals > 1e-8
        supp_vecs = evecs[:, mask]
        
        dim = rho_dm.data.shape[0]
        if supp_vecs.shape[1] > 0:
            proj = supp_vecs @ supp_vecs.conj().T
            proj = proj / np.trace(proj) # Normalize trace to 1
            return DensityMatrix(proj)
            
        return DensityMatrix(np.zeros((dim, dim), dtype=complex))

    def _intersect(self, matrices, target_size):
        """Finds the intersection of expanded subspaces."""
        if not matrices: 
            return np.zeros((target_size, target_size), dtype=complex)
        
        M = len(matrices) * np.eye(target_size, dtype=complex)
        for mat in matrices:
            evals, evecs = np.linalg.eigh(mat)
            mask = evals > 1e-8
            vecs = evecs[:, mask]
            proj = vecs @ vecs.conj().T if vecs.shape[1] > 0 else np.zeros_like(mat)
            M -= proj
            
        ns = null_space(M)
        if ns.shape[1] == 0:
            return np.zeros((target_size, target_size), dtype=complex)
        
        res = ns @ ns.conj().T
        return res / np.trace(res)

    def process_circuit(self, qc):
        dag = circuit_to_dag(qc)
        for node in dag.topological_op_nodes():
            gate = node.op
            if gate.name in ['measure', 'barrier']:
                continue
            
            qubits = tuple(qc.find_bit(q).index for q in node.qargs)
            self._apply_sandwich(gate, qubits)

    def _apply_sandwich(self, gate, F):
        """Expand -> Operate -> Compress"""
        F_set = set(F)
        gate_op = Operator(gate)
        
        # Optimize: Only process subsets overlapping with the gate
        affected_subsets = [s for s in self.domain if set(s).intersection(F_set)]
        new_states_for_affected = {}
        
        for s_i in affected_subsets:
            T_i = tuple(sorted(set(s_i).union(F_set)))
            covered_subsets = [p for p in self.domain if set(p).issubset(T_i)]
            expanded_matrices =[]
            
            for p in covered_subsets:
                local_dm = self.state[p]
                
                # Expand dynamically via SparsePauliOp
                local_pauli = SparsePauliOp.from_operator(local_dm.data)
                full_op = Operator(np.eye(2**len(T_i)))
                q_indices =[T_i.index(q) for q in p]
                
                expanded_mat = full_op.compose(local_pauli, qargs=q_indices).data
                expanded_matrices.append(expanded_mat)
            
            if len(T_i) > len(s_i):
                Q_Ti_data = self._intersect(expanded_matrices, 2**len(T_i))
            else:
                Q_Ti_data = expanded_matrices[0]
                
            Q_Ti_dm = DensityMatrix(Q_Ti_data)
            
            # Operate
            gate_indices =[T_i.index(q) for q in F]
            Q_Ti_prime = Q_Ti_dm.evolve(gate_op, qargs=gate_indices)
            
            # Compress
            trace_indices =[T_i.index(q) for q in T_i if q not in s_i]
            if trace_indices:
                rho_reduced = partial_trace(Q_Ti_prime, trace_indices)
            else:
                rho_reduced = Q_Ti_prime
                
            new_states_for_affected[s_i] = self._get_support(rho_reduced)
            
        # Broadcast Updates
        for s_i, new_dm in new_states_for_affected.items():
            self.state[s_i] = new_dm

# ==========================================
# 3. DYNAMIC ASSERTION CHECKING
# ==========================================
def build_ghz_circuit(n):
    """Star-Topology GHZ circuit"""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    return qc

def get_ghz_projector_for_k_qubits(k):
    """Dynamically generates the SparsePauliOp Projector for span{|0..0>, |1..1>}"""
    size = 2**k
    proj_mat = np.zeros((size, size), dtype=complex)
    proj_mat[0, 0] = 1.0           # |00...0><00...0|
    proj_mat[size-1, size-1] = 1.0 # |11...1><11...1|
    return SparsePauliOp.from_operator(proj_mat)

def run_qiskit_native_benchmark():
    qubit_sizes =[3, 10, 30, 60, 300] 
    
    print(f"{'Qubits':<8} | {'Subsets Tracked':<15} | {'Analysis Time':<15} | {'Status'}")
    print("-" * 65)
    
    for n in qubit_sizes:
        qc = build_ghz_circuit(n)
        
        # 1. Dynamically Generate Domain based on Circuit Flow
        domain = generate_domain_from_circuit(qc)
        
        # 2. Process Circuit
        start_time = time.perf_counter()
        analyzer = AbstractState(qc, domain)
        analyzer.process_circuit(qc)
        t_analysis = time.perf_counter() - start_time
        
        # 3. Verify Assertions
        is_verified = True
        
        for subset in analyzer.domain:
            k = len(subset) # Get specific size of this tracked subset
            pauli_projector = get_ghz_projector_for_k_qubits(k)
            local_dm = analyzer.state[subset]
            
            exp_val = np.real(local_dm.expectation_value(pauli_projector))
            
            if not np.isclose(exp_val, 1.0, atol=1e-5):
                is_verified = False
                break
                
        status = "✅ PASS" if is_verified else "❌ FAIL"
        subsets_tracked = f"{len(analyzer.domain):,}"
        
        print(f"{n:<8} | {subsets_tracked:<15} | {t_analysis:<11.4f} sec | {status}")

if __name__ == "__main__":
    print("Starting Data-Flow Optimized Abstract Interpretation...")
    run_qiskit_native_benchmark()