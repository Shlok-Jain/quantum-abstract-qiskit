from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# 1. Create a circuit with 3 qubits and 3 classical bits
qc = QuantumCircuit(3, 3)

# 2. Add a Hadamard gate on qubit 0 to create superposition (|0> + |1>) / sqrt(2)
qc.h(0)

# 3. Add CNOT gates to entangle qubits 1 and 2 with qubit 0
qc.cx(0, 1)
qc.cx(0, 2)

# 4. Measure all qubits into the classical bits
qc.measure([0, 1, 2], [0, 1, 2])

# 5. Simulate the circuit using the AerSimulator
simulator = AerSimulator()
result = simulator.run(qc).result()

# 6. Get and print the counts (expecting ~50% '000' and ~50% '111')
counts = result.get_counts()
print(f"Measurement counts: {counts}")

# (Optional) Draw the circuit
# print(qc.draw())
