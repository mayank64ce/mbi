import numpy as np
from openfhe import *
import openfhe_numpy as onp


def test_vector_metadata():
    """
    Test accessing vector and matrix metadata without decryption.
    This demonstrates: 
    - Getting vector length from ciphertext
    - Getting matrix dimensions from ciphertext
    - Comparing with plaintext values
    """
    print("=" * 80)
    print("TEST:  ACCESSING ENCRYPTED VECTOR/MATRIX METADATA WITHOUT DECRYPTION")
    print("=" * 80)

    # ===== CRYPTOGRAPHIC SETUP =====
    print("\n[1] Initializing cryptographic context...")
    mult_depth = 4
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys. secretKey)
    cc.EvalSumKeyGen(keys. secretKey)

    ring_dim = cc.GetRingDimension()
    batch_size = ring_dim // 2
    print(f"    Ring dimension: {ring_dim}")
    print(f"    Available slots (batch size): {batch_size}")

    # ===== TEST 1: VECTOR METADATA =====
    print("\n" + "-" * 80)
    print("TEST 1: Vector Metadata Access")
    print("-" * 80)

    vector_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\nPlaintext vector:  {vector_a}")
    print(f"Plaintext vector length: {len(vector_a)}")

    # Encrypt the vector
    ct_vector_a = onp.array(
        cc=cc,
        data=vector_a,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys. publicKey,
    )
    print(f"\nEncrypted vector created (ciphertext)")

    # ===== GET METADATA WITHOUT DECRYPTION =====
    print("\n[Accessing metadata WITHOUT decryption]")
    
    # Shape
    shape = ct_vector_a.shape
    print(f"    ct_vector. shape: {shape}")
    
    # Original shape (before padding)
    original_shape = ct_vector_a.original_shape
    print(f"    ct_vector.original_shape: {original_shape}")
    
    # Number of dimensions
    ndim = ct_vector_a.ndim
    print(f"    ct_vector.ndim: {ndim}")
    
    # Vector length
    vector_length = ct_vector_a.shape[0]
    print(f"    ct_vector.shape[0] (LENGTH): {vector_length}")
    
    # Batch size
    batch_sz = ct_vector_a.batch_size
    print(f"    ct_vector.batch_size: {batch_sz}")
    
    # Order
    order = ct_vector_a.order
    print(f"    ct_vector.order: {order}")

    # ===== VERIFY WITH DECRYPTION =====
    print("\n[Verifying with decryption]")
    decrypted_a = ct_vector_a.decrypt(keys.secretKey, unpack_type="original")
    print(f"    Decrypted vector: {decrypted_a}")
    print(f"    Decrypted length: {len(decrypted_a)}")
    
    # Check if metadata matches actual data
    match = vector_length == len(decrypted_a)
    print(f"    Metadata length matches decrypted length: {match}")
    
    # Check values match
    values_match = np.allclose(decrypted_a, vector_a)
    print(f"    Decrypted values match plaintext: {values_match}")

    # # ===== TEST 2: MATRIX METADATA =====
    # print("\n" + "-" * 80)
    # print("TEST 2: Matrix Metadata Access")
    # print("-" * 80)

    # matrix_b = np.array([
    #     [1.0, 2.0, 3.0],
    #     [4.0, 5.0, 6.0],
    #     [7.0, 8.0, 9.0]
    # ])
    # print(f"\nPlaintext matrix:\n{matrix_b}")
    # print(f"Plaintext shape: {matrix_b.shape}")

    # # Encrypt the matrix
    # ct_matrix_b = onp.array(
    #     cc=cc,
    #     data=matrix_b,
    #     batch_size=batch_size,
    #     order=onp.ROW_MAJOR,
    #     mode="zero",
    #     fhe_type="C",
    #     public_key=keys. publicKey,
    # )
    # print(f"Encrypted matrix created (ciphertext)")

    # # ===== GET MATRIX METADATA WITHOUT DECRYPTION =====
    # print("\n[Accessing matrix metadata WITHOUT decryption]")
    
    # # Shape
    # matrix_shape = ct_matrix_b.shape
    # print(f"    ct_matrix.shape: {matrix_shape}")
    
    # # Original shape
    # matrix_original_shape = ct_matrix_b.original_shape
    # print(f"    ct_matrix.original_shape: {matrix_original_shape}")
    
    # # Dimensions
    # matrix_ndim = ct_matrix_b.ndim
    # print(f"    ct_matrix.ndim: {matrix_ndim}")
    
    # # Rows and columns
    # matrix_rows = ct_matrix_b.shape[0]
    # matrix_cols = ct_matrix_b.shape[1]
    # print(f"    ct_matrix.shape[0] (ROWS): {matrix_rows}")
    # print(f"    ct_matrix.shape[1] (COLS): {matrix_cols}")
    
    # # Alternative ways to get dimensions
    # ncols = ct_matrix_b. ncols
    # nrows = ct_matrix_b.nrows
    # print(f"    ct_matrix.ncols: {ncols}")
    # print(f"    ct_matrix.nrows: {nrows}")
    
    # # Batch size
    # matrix_batch_sz = ct_matrix_b.batch_size
    # print(f"    ct_matrix.batch_size: {matrix_batch_sz}")

    # # ===== VERIFY MATRIX WITH DECRYPTION =====
    # print("\n[Verifying matrix with decryption]")
    # decrypted_b = ct_matrix_b.decrypt(keys.secretKey, unpack_type="original")
    # print(f"    Decrypted matrix shape: {decrypted_b. shape}")
    # print(f"    Decrypted matrix:\n{decrypted_b}")
    
    # # Check if metadata matches
    # shape_match = (matrix_rows, matrix_cols) == decrypted_b.shape
    # print(f"    Metadata shape matches decrypted shape: {shape_match}")
    
    # # Check values match
    # matrix_values_match = np.allclose(decrypted_b, matrix_b)
    # print(f"    Decrypted values match plaintext: {matrix_values_match}")

    # # ===== TEST 3: VECTOR WITH DIFFERENT PADDING MODE =====
    # print("\n" + "-" * 80)
    # print("TEST 3: Vector with Tiling Mode")
    # print("-" * 80)

    # vector_c = np.array([1.1, 2.2, 3.3])
    # print(f"\nPlaintext vector: {vector_c}")

    # ct_vector_c = onp. array(
    #     cc=cc,
    #     data=vector_c,
    #     batch_size=batch_size,
    #     order=onp.ROW_MAJOR,
    #     mode="tile",  # Using tile mode instead of zero padding
    #     fhe_type="C",
    #     public_key=keys.publicKey,
    # )

    # print(f"Encrypted with 'tile' mode")
    # print(f"\n[Metadata WITHOUT decryption]")
    # print(f"    ct_vector.shape: {ct_vector_c.shape}")
    # print(f"    ct_vector.original_shape: {ct_vector_c.original_shape}")
    # print(f"    ct_vector.batch_size: {ct_vector_c.batch_size}")

    # print(f"\n[Verifying with decryption]")
    # decrypted_c = ct_vector_c.decrypt(keys.secretKey, unpack_type="original")
    # print(f"    Decrypted vector (original values only): {decrypted_c}")
    # print(f"    Decrypted length: {len(decrypted_c)}")
    # print(f"    Match with plaintext: {np.allclose(decrypted_c, vector_c)}")

    # # ===== SUMMARY =====
    # print("\n" + "=" * 80)
    # print("SUMMARY")
    # print("=" * 80)
    # print("\n✓ Vector/matrix metadata can be accessed WITHOUT decryption:")
    # print("    - shape / original_shape")
    # print("    - ndim")
    # print("    - batch_size")
    # print("    - ncols / nrows")
    # print("    - order")
    # print("\n✓ Only the actual encrypted VALUES require decryption")
    # print("\n✓ Metadata is stored as plaintext in the CTArray wrapper")
    # print("=" * 80)


if __name__ == "__main__":
    test_vector_metadata()