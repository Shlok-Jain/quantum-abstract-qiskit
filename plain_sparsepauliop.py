import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator

def get_ghz_circuit(n, theta=np.pi/4):
    """Generates a GHZ circuit for n qubits."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        # Replaced the Clifford 'cx' with a non-Clifford 'crx' (Controlled-RX)
        qc.crx(theta, i, i + 1)
    return qc

def get_ghz_assertions(n):
    """
    Creates local assertions using SparsePauliOp.
    Following the paper, we check local pairs. 
    A pair is in span(|00>, |11>) if the expectation of 0.5(II + ZZ) is 1.0.
    """
    assertions = []
    for i in range(n - 1):
        # Create a ZZ operator for qubits i and i+1
        # 'I' * (n) creates the base identity string
        pauli_list = []
        
        # Identity term (1/2 * II)
        pauli_list.append(("I" * n, 0.5))
        
        # ZZ term (1/2 * ZZ)
        z_string = list("I" * n)
        z_string[i] = "Z"
        z_string[i+1] = "Z"
        pauli_list.append(("".join(z_string), 0.5))
        
        assertions.append(SparsePauliOp.from_list(pauli_list))
    return assertions

def verify_ghz(n):
    """Runs the circuit and verifies all local assertions."""
    start_time = time.time()
    
    # 1. Build components
    qc = get_ghz_circuit(n)
    ops = get_ghz_assertions(n)
    
    # 2. Use Aer Estimator for high-performance verification
    # For GHZ, Aer uses Matrix Product States which is very fast
    estimator = Estimator()
    
    # 3. Run estimation
    # We pass the circuit once and a list of all local operators
    job = estimator.run([qc] * len(ops), ops)
    results = job.result()
    
    # 4. Check results (all must be approx 1.0)
    all_passed = np.allclose(results.values, 1.0)
    
    end_time = time.time()
    return all_passed, end_time - start_time

# --- Testing Loop ---
test_ns = [3, 30, 60, 90]
print(f"{'n':<10} | {'Status':<10} | {'Time (s)':<15}")
print("-" * 40)

for n in test_ns:
    try:
        passed, duration = verify_ghz(n)
        status = "PASSED" if passed else "FAILED"
        print(f"{n:<10} | {status:<10} | {duration:<15.4f}")
    except Exception as e:
        print(f"{n:<10} | ERROR      | {str(e)[:20]}...")