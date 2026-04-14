import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_static_analyzer import QuantumStaticAnalyzer

def generate_domain_from_circuit(circuit: QuantumCircuit) -> list[tuple[int, ...]]:
    """
    Automatically extracts the necessary domain subsets by tracking all 
    multi-qubit interactions in the circuit (similar to the Java heuristic).
    """
    domain_set = set()
    for instruction in circuit.data:
        # Ignore barriers and measurements
        if instruction.operation.name in ['barrier', 'measure', 'reset', 'delay']:
             continue
        if len(instruction.qubits) >= 2:
            qubit_indices = tuple(sorted(circuit.find_bit(q).index for q in instruction.qubits))
            domain_set.add(qubit_indices)
            
    # Fallback for circuits with purely single-qubit gates
    if not domain_set:
        domain_set = {(circuit.find_bit(q).index,) for q in circuit.qubits}
        
    return list(domain_set)

def analyze_circuit(circuit: QuantumCircuit, domain_subsets: list[tuple[int, ...]] | None = None):
    """
    Takes a Qiskit QuantumCircuit and automatically applies the Quantum Static Analyzer
    to all its quantum operations, step by step.
    
    Args:
        circuit: A Qiskit QuantumCircuit mapped to standard integer registers.
        domain_subsets: The tuples of qubits to track abstractly. (Auto-infers if None)
        
    Returns:
        The updated QuantumStaticAnalyzer instance containing the final state.
    """
    # Auto-generate domain if not provided
    if domain_subsets is None:
        domain_subsets = generate_domain_from_circuit(circuit)
        
    analyzer = QuantumStaticAnalyzer(domain_subsets)
    
    # Iterate over every instruction in the quantum circuit exactly in order
    for instruction in circuit.data:
        op = instruction.operation
        
        # We skip classical operations
        if op.name in ['barrier', 'measure', 'reset']:
            continue
            
        # Get the unitary matrix representation of the gate
        try:
            gate_matrix = Operator(op).data
        except Exception as e:
            print(f"Skipping non-unitary operation '{op.name}': {e}")
            continue
            
        # Extract the integer indices of the qubits this gate acts upon
        qubit_indices = [circuit.find_bit(q).index for q in instruction.qubits]
        
        # The analyzer expects gate matrices and a list of target integer indices
        analyzer.apply_gate(gate_matrix, qubit_indices)
        
    return analyzer
    
# Example integration matching your ghz003.py circuit structure
if __name__ == "__main__":
    import time
    
    def stress_test(num_qubits):
        print(f"\n{'='*40}\nStress Testing GHZ for {num_qubits} qubits\n{'='*40}")
        
        # 1. GHZ Circuit Construction
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for i in range(num_qubits-1):
            qc.cx(i, i+1)
            
        # --- Run Quantum Abstract Interpretation ---
        # 2. Automatically extract domain by passing None
        start_time = time.time()
        final_analyzer = analyze_circuit(qc)  # NO manual domain passed
        domain = final_analyzer.domain
        analyze_time = time.time() - start_time
        print(f"[*] Static Analysis compiled in {analyze_time:.4f} seconds.")
        
        # 3. The 'Final Verdict' logic adapted for scale:
        # Building a 2^300 length vector for the global spanning state |00...0> will crash the OS natively. 
        # But we mathematically know GHZ's universal global projection forces EVERY abstract pair 
        # to exist identically in span{|00>, |11>}.
        subspace_state00 = np.array([1, 0, 0, 0])
        subspace_state11 = np.array([0, 0, 0, 1])
        # P = |00><00| + |11><11|
        assert_projector = np.outer(subspace_state00, subspace_state00) + np.outer(subspace_state11, subspace_state11)
        
        verdict = True
        for s_i in domain:
            if not final_analyzer.check_assertion(assert_projector, s_i):
                verdict = False
                break
                
        print(f"[*] Final Global Verdict: {verdict}")
        return analyze_time

    qubit_sizes = [3, 30, 60, 90, 300]
    results = []
    
    for n in qubit_sizes:
        t = stress_test(n)
        results.append((n, t))
        
    print("\n--- Benchmark Summary ---")
    for n, t in results:
        print(f"{n:>3} qubits | Time: {t:.4f} sec")
