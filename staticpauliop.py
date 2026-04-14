import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import StabilizerState, SparsePauliOp

def get_ghz_assertions(n):
    """
    Here is where YOU state that it must be span{|00...>, |11...>}.
    We manually define the Pauli strings that represent this geometry.
    """
    assertions =[]
    
    # 1. Parity rules: Adjacent qubits must be identical (Checks it's 000... or 111...)
    for i in range(n - 1):
        # Create an array of 'I's
        pauli_list = ['I'] * n
        # Put 'Z' on adjacent qubits i and i+1
        pauli_list[i] = 'Z'
        pauli_list[i+1] = 'Z'
        
        # Join into a string (Reverse because Qiskit reads right-to-left)
        pauli_string = "".join(pauli_list)[::-1]
        assertions.append(SparsePauliOp.from_list([(pauli_string, 1.0)]))
        
    # 2. Phase rule: Must be the specific superposition (Checks the '+' in the span)
    x_string = "X" * n
    assertions.append(SparsePauliOp.from_list([(x_string, 1.0)]))
    
    return assertions

# --- Let's test it ---
N_QUBITS = 300

# 1. The Implementation (The Circuit)
qc = QuantumCircuit(N_QUBITS)
qc.h(0)
for i in range(1, N_QUBITS):
    # qc.cx(0, i)
    qc.crx(np.pi/4, 0, i)

# 2. The Specification (Your explicit assertions)
my_assertions = get_ghz_assertions(N_QUBITS)

# 3. Simulate and Verify
final_state = StabilizerState(qc)

print("Testing if state is in span{|000...>, |111...>}...\n")
is_verified = True

for op in my_assertions:
    exp_val = np.real(final_state.expectation_value(op))
    print(f"Rule {op.paulis[0]}: Expectation = {exp_val}")
    if not np.isclose(exp_val, 1.0):
        is_verified = False

if is_verified:
    print("\n✅ VERIFIED: The state exactly matches your assertion!")
else:
    print("\n❌ FAILED: The state does not match.")