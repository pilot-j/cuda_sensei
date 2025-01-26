
# CUDA C Basics

This folder contains implementations of basic CUDA programming concepts. These exercises aim to introduce the foundations of GPU programming using CUDA, focusing on thread management, memory utilization, and performance optimization. I have been following the flow of *CUDA by example* and highly recommend that book to absolute beginners like me. 
The idea is to build up to a level where we can write kernels that are relevant to and used in Deep Learning Networks (e.g., Norms, Activations, Attention, etc.).

---

## **Exercises**

### 1. **Vector Addition**
**Description:**  
This program demonstrates the use of multiple threads for vector addition. It showcases how to handle very large vectors by reusing threads, dividing the workload across the GPU efficiently.

**Key Concepts:**  
- Launching a large number of threads using CUDA.  
- Reusing threads to handle vectors larger than the total number of threads.  
- Efficient memory utilization for parallel computation.  

**File:** `vec_addition.cu`

---

### 2. **Dot Product**
**Description:**  
This program computes the dot product of two vectors using the concept of shared memory and reduction for performance optimization. Care is taken to avoid divergent code branches to ensure proper program execution.

**Key Concepts:**  
- **Shared Memory:** Used to store intermediate results within each thread block, reducing the number of global memory accesses.  
- **Reduction:** Gradually combines partial results from threads to compute the final output efficiently.  
- **Divergent Branches:** Highlights the potential issues when using conditionals (`if` statements) with `__syncthreads()`.  
    - **Caution:** All threads within a block must execute the `__syncthreads()` command; otherwise, the program may hang indefinitely. Divergent branches occur when only a subset of threads execute an `if` block, causing others to miss the synchronization point.

**File:** `dot_product.cu`

---

## **How to Compile and Run**
1. Ensure you have the CUDA toolkit installed.
2. Use the `nvcc` compiler to compile the programs:
   ```bash
   nvcc vector_addition.cu -o vector_addition
   nvcc dot_product.cu -o dot_product
   ```
3. Run the compiled executables:
   ```bash
   ./vector_addition
   ./dot_product
   ```


Happy Coding! 🚀
