import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, partial_trace

def get_expanded_operator(P_data, current_qubits, target_qubits):
    """
    Step 1 Helper: Focus/Concretization (gamma)
    Expands operator P_data (acting on current_qubits) to target_qubits
    by tensoring with Identity on the missing qubits.
    """
    mapping = [target_qubits.index(q) for q in current_qubits]
    
    # Start with a full Identity operator on the target number of qubits
    I_full = Operator(np.eye(2 ** len(target_qubits)))
    
    if len(mapping) > 0:
        # compose the operator onto the target indices
        # This mathematically tensors P_data with Identity without requiring Unitarity
        result_op = I_full.compose(Operator(P_data), qargs=mapping)
        return result_op.data
        
    return I_full.data

def abstract_projection(Q_data, current_qubits, keep_qubits, atol=1e-8):
    """
    Step 3: Abstraction (alpha)
    Traces out unwanted qubits from Q_data and returns the support projection.
    """
    trace_qubits = [q for q in current_qubits if q not in keep_qubits]
    
    # Convert qubits to trace into local indices
    trace_indices = [current_qubits.index(q) for q in trace_qubits]
    
    # 1. Partial Trace
    if len(trace_indices) > 0:
        rho_reduced = partial_trace(Q_data, trace_indices).data
    else:
        rho_reduced = Q_data

    # 2. Support computation (eigenvectors with non-zero eigenvalues)
    evals, evecs = np.linalg.eigh(rho_reduced)
    mask = evals > atol
    non_zero_evecs = evecs[:, mask]
    
    if non_zero_evecs.shape[1] == 0:
        return np.zeros_like(rho_reduced)
        
    # Projector P = sum |v><v|
    return non_zero_evecs @ non_zero_evecs.conj().T

def intersect_projections(projections):
    """
    Intersection: Combines multiple overlapping abstract states.
    Uses the null space of sum (I - P_i).
    I want that a vector v is in the intersection if and only if 
    P_i v = v for all i.
    This is equivalent to 
    (I - P_i) v = 0 for all i
    So, I have to find null space of sum(I - P_i) across all i.
    """
    if not projections:
        return None
    if len(projections) == 1:
        return projections[0]
        
    n = projections[0].shape[0]
    M = len(projections) * np.eye(n, dtype=complex)
    for P in projections:
        M -= P
        
    # Vectors strictly preserved by all P_i are in the null space
    ns = scipy.linalg.null_space(M)
    if ns.shape[1] == 0:
        return np.zeros((n, n), dtype=complex)
    return ns @ ns.conj().T

class QuantumStaticAnalyzer:
    def __init__(self, domain_subsets):
        """
        domain_subsets: list of tuples of qubit indices, e.g. [(0, 1), (1, 2)]
        """
        # Ensure subsets are sorted for consistency
        self.domain = [tuple(sorted(s)) for s in domain_subsets]
        self.state = {}
        
        # Initialize each subset with computational basis projector |0..0><0..0|
        for s in self.domain:
            size = 2 ** len(s)
            P = np.zeros((size, size), dtype=complex)
            P[0, 0] = 1.0
            self.state[s] = P
            
    def apply_gate(self, u_data, gate_qubits):
        """
        Applies a unitary gate using the Sandwich Method (Focus -> Operate -> Abstract)
        u_data: Unitary matrix (numpy array or Qiskit Operator)
        gate_qubits: Qubits the gate acts on (e.g. F)
        """
        F = tuple(sorted(gate_qubits))
        F_set = set(F)
        new_state = {}
        
        for s_i in self.domain:
            # Optimize: Only process subsets overlapping with the gate
            if not set(s_i).intersection(F_set):
                new_state[s_i] = self.state[s_i]
                continue
                
            # T_i = s_i U F (Target finer domain)
            T_i = tuple(sorted(set(s_i).union(F_set)))
            
            # Find all subsets s_j that are completely covered by T_i
            # We expand all such s_j to T_i and intersect them later.
            covered_subsets = [s_j for s_j in self.domain if set(s_j).issubset(T_i)]
            
            # --- 1. Focus / Concretization ---
            expanded_projections = []
            for s_j in covered_subsets:
                P_j = self.state[s_j]
                Q_j = get_expanded_operator(P_j, s_j, T_i)
                expanded_projections.append(Q_j)
                
            # Intersect overlapping knowledge in the expanded domain T_i
            Q_Ti = intersect_projections(expanded_projections)
            
            # --- 2. Concrete Operation ---
            # Expand the Unitary U to act on the full T_i
            U_expanded = get_expanded_operator(u_data, F, T_i)
            # Apply unitary to the projection: Q' = U Q U^dagger
            Q_Ti_prime = U_expanded @ Q_Ti @ U_expanded.conj().T
            
            # --- 3. Abstraction ---
            # Compress back to original subset s_i
            P_i_prime = abstract_projection(Q_Ti_prime, T_i, s_i)
            
            new_state[s_i] = P_i_prime
            
        # Update the global state
        self.state = new_state

    def check_assertion(self, assert_matrix, target_qubits):
        """
        Verifies if the current state sits inside an assertion subspace.
        assert_matrix: Projection matrix of the assertion
        """
        assert_qubits = tuple(sorted(target_qubits))
        if assert_qubits not in self.domain:
             raise ValueError("Assertion qubits must exactly match one of the tracked subsets in the domain.")
             
        # Extract the current state projection P_final
        P_final = self.state[assert_qubits]
        
        # Check if supp(P_final) is a subspace of supp(P_assert)
        # This is true if P_assert * P_final == P_final
        return np.allclose(assert_matrix @ P_final, P_final, atol=1e-5)

    def check_global_assertion(self, global_states, all_qubits):
        """
        Takes a list of global states (representing the span of the assertion)
        defined on `all_qubits`. Checks if ALL local tracked subsets conform
        to this global assertion by automatically projecting the global states 
        down to each subset domain.
        Returns True (valid) or False (invalid).
        """
        all_qubits = tuple(sorted(all_qubits))
        
        # Form the global density matrix representing the uniform mixture of spanning states
        rho_global = sum(np.outer(v, v.conjugate()) for v in global_states)
        
        for s_i in self.domain:
            # Trace out (all_qubits \ s_i) from rho_global
            trace_qubits = [q for q in all_qubits if q not in s_i]
            trace_indices = [all_qubits.index(q) for q in trace_qubits]
            
            if len(trace_indices) > 0:
                rho_local = partial_trace(rho_global, trace_indices).data
            else:
                rho_local = rho_global
                
            # The local assertion projector is the support of this reduced density matrix
            from numpy.linalg import eigh
            evals, evecs = eigh(rho_local)
            mask = evals > 1e-8
            non_zero_evecs = evecs[:, mask]
            
            if non_zero_evecs.shape[1] == 0:
                local_assert_projector = np.zeros_like(rho_local)
            else:
                local_assert_projector = non_zero_evecs @ non_zero_evecs.conj().T
                
            # Now check the local abstract state
            P_final = self.state[s_i]
            
            # If P_final is NOT a subspace of local_assert_projector, it fails early
            if not np.allclose(local_assert_projector @ P_final, P_final, atol=1e-5):
                return False
                
        # If all tracked subsets conform to the global assertion projections:
        return True

# Example Usage:
if __name__ == "__main__":
    from qiskit.circuit.library import CXGate, HGate
    
    # 1. Define the tuple of qubit subsets S = (s1, s2)
    analyzer = QuantumStaticAnalyzer([(0, 1), (1, 2)])
    
    # 2. Apply Hadamard on Qubit 0
    analyzer.apply_gate(HGate().to_matrix(), [0])
    
    # 3. Apply CNOT on Qubits 0 and 1
    analyzer.apply_gate(CXGate().to_matrix(), [0, 1])

    # 4. Check an Assertion
    # We assert that qubits (0, 1) are in the Bell state |00> + |11>
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    bell_projector = np.outer(bell_state, bell_state.conj())
    
    is_valid = analyzer.check_assertion(bell_projector, [0, 1])
    print("Is the assertion valid? ->", is_valid)
