import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_inbuilt_analyzer import analyze_circuit

def test_parametrized_rotation(num_qubits=3):
    """
    Tests the static analyzer on a circuit where the first qubit is rotated 
    by an angle theta, then entangled.
    
    Resulting State: cos(theta/2)|00...0> + sin(theta/2)|11...1>
    """
    print(f"\n{'='*60}")
    print(f"Testing Parametrized GHZ-Rotation for {num_qubits} qubits")
    print(f"{'='*60}\n")

    # 1. Define the Parametrized Circuit
    theta_param = Parameter('θ')
    qc_template = QuantumCircuit(num_qubits)
    qc_template.ry(theta_param, 0)
    for i in range(num_qubits - 1):
        qc_template.cx(i, i + 1)

    # 2. Define angles to test (in radians)
    # We test: 0, Pi/4 (45°), Pi/2 (90°), 3Pi/4 (135°), Pi (180°)
    test_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    all_results_pass = True

    for angle in test_angles:
        print(f"--- Analyzing Configuration: θ = {angle:.4f} rad ({np.degrees(angle):.1f}°) ---")
        
        # 3. Bind the parameter to a specific value
        bound_qc = qc_template.assign_parameters({theta_param: angle})
        
        # 4. Generate the Abstract State via Static Analysis
        # This will auto-extract domains like (0,1), (1,2), etc.
        analyzer = analyze_circuit(bound_qc)
        
        # 5. Generate the expected "Ground Truth" vector for assertion
        # Mathematically: |psi> = cos(theta/2)|0...0> + sin(theta/2)|1...1>
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        
        # Construct the 2^n vector
        vec_000 = np.zeros(2**num_qubits)
        vec_000[0] = 1.0
        
        vec_111 = np.zeros(2**num_qubits)
        vec_111[-1] = 1.0
        
        expected_global_state = c * vec_000 + s * vec_111
        
        # 6. Verify Assertion
        # check_global_assertion takes a list of vectors representing the span
        # of the allowed subspace.
        is_valid = analyzer.check_global_assertion([expected_global_state], range(num_qubits))
        
        status = "PASSED" if is_valid else "FAILED" 
        print(f"Result: {status}")
        if not is_valid:
            all_results_pass = False

    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {'ALL TESTS PASSED' if all_results_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")

def test_parametrized_multi_variable():
    """
    Tests a circuit with multiple independent parameters.
    """
    print(f"\nStarting Multi-Parameter Test (2 Qubits)...")
    a = Parameter('α')
    b = Parameter('β')
    qc = QuantumCircuit(2)
    qc.rx(a, 0)
    qc.ry(b, 1)
    qc.cx(0, 1)

    # Test values
    val_a, val_b = np.pi/3, np.pi/6
    bound_qc = qc.assign_parameters({a: val_a, b: val_b})
    
    # Run Static Analysis
    analyzer = analyze_circuit(bound_qc)
    
    # Generate ground truth for assertion using exact matrix math
    from qiskit.quantum_info import Statevector
    actual_vector = Statevector.from_instruction(bound_qc).data
    
    # Verify
    is_valid = analyzer.check_global_assertion([actual_vector], [0, 1])
    print(f"Multi-variable (α={val_a:.2f}, β={val_b:.2f}) result: {'PASSED' if is_valid else 'FAILED'}")

if __name__ == "__main__":
    # Test 1: Scaling angle on GHZ
    test_parametrized_rotation(num_qubits=4)
    
    # Test 2: Multiple different parameters
    test_parametrized_multi_variable()