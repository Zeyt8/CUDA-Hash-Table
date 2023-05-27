#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
*/

__device__ static int fnvHash(const char* str)
{
    unsigned int hash = 2166136261u;
	hash ^= *(str + 0);
    hash *= 16777619u;
	hash ^= *(str + 1);
    hash *= 16777619u;
	hash ^= *(str + 2);
    hash *= 16777619u;
	hash ^= *(str + 3);
    hash *= 16777619u;
	hash &= 0x7FFFFFFF;
    return (int)hash;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) {
	glbGpuAllocator->_cudaMalloc((void**)&table, size * sizeof(HashTableItem));
	cudaMemset(table, 0, size * sizeof(HashTableItem));
	capacity = size;
	count = 0;
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(table);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
__global__ void reshapeKernel(HashTableItem* newTable, HashTableItem* table, int size, int numBucketsReshape)
{
	// calculate index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// if index is out of bounds, return
	if (index >= size) {
		return;
	}
	// if previous entry is empty, return
	if (table[index].key == 0) {
		return;
	}
	// recalculate hash
	int hash = fnvHash((char*)&table[index].key) % numBucketsReshape;
	// find place to insert
	while (true) {
		int prev = atomicCAS(&newTable[hash].key, 0, table[index].key);
		if (prev == 0) {
			newTable[hash].value = table[index].value;
			return;
		}
		hash = (hash + 1) % numBucketsReshape;
	}
}

void GpuHashTable::reshape(int numBucketsReshape) {
	// alloc new table
	HashTableItem* newTable;
	glbGpuAllocator->_cudaMalloc((void**)&newTable, numBucketsReshape * sizeof(HashTableItem));
	cudaMemset(newTable, 0, numBucketsReshape * sizeof(HashTableItem));
	// call kernel
	reshapeKernel<<<(capacity + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(newTable, table, capacity, numBucketsReshape);
	cudaDeviceSynchronize();
	// update table
	glbGpuAllocator->_cudaFree(table);
	table = newTable;
	capacity = numBucketsReshape;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
__global__ void insertBatchKernel(HashTableItem* table, int size, int* keys, int* values, int numKeys, int* added)
{
	// calculate index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// if index is out of bounds, return
	if (index >= numKeys) {
		return;
	}
	// calculate hash
	int key = keys[index];
	int value = values[index];
	int hash = fnvHash((char*)&key) % size;
	// find place to insert
	while (true) {
		int prev = atomicCAS(&table[hash].key, 0, key);
		if (prev == 0) {
			// entry did not exist
			table[hash].value = value;
			atomicAdd(added, 1);
			return;
		}
		else if (prev == key) {
			// entry already exists
			table[hash].value = value;
			return;
		}
		hash = (hash + 1) % size;
	}
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	// check if we need to reshape
	if (count + numKeys > capacity * LOAD_FACTOR) {
		int newSize = capacity;
		while (count + numKeys > newSize * LOAD_FACTOR) {
			newSize *= 2;
		}
		reshape(newSize);
	}
	// move keys and values to GPU
	int* keysDevice;
	glbGpuAllocator->_cudaMalloc((void**)&keysDevice, numKeys * sizeof(int));
	cudaMemcpy(keysDevice, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	int* valuesDevice;
	glbGpuAllocator->_cudaMalloc((void**)&valuesDevice, numKeys * sizeof(int));
	cudaMemcpy(valuesDevice, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	// keep track of how many keys were actually added
	int* added;
	glbGpuAllocator->_cudaMalloc((void**)&added, sizeof(int));
	cudaMemset(added, 0, sizeof(int));
	// call kernel
	insertBatchKernel<<<(numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(table, capacity, keysDevice, valuesDevice, numKeys, added);
	cudaDeviceSynchronize();
	// update count
	int* addedHost = (int*)malloc(sizeof(int));
	cudaMemcpy(addedHost, added, sizeof(int), cudaMemcpyDeviceToHost);
	count += *addedHost;
	// cleanup
	free(addedHost);
	glbGpuAllocator->_cudaFree(added);
	glbGpuAllocator->_cudaFree(keysDevice);
	glbGpuAllocator->_cudaFree(valuesDevice);
	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
__global__ void getBatchKernel(HashTableItem* table, int size, int* keys, int* values, int numKeys)
{
	// calculate index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// if index is out of bounds, return
	if (index >= numKeys) {
		return;
	}
	// calculate hash
	int key = keys[index];
	int hash = fnvHash((char*)&key) % size;
	int start = hash;
	// find place to insert
	while (true) {
		if (table[hash].key == key) {
			// entry exists
			values[index] = table[hash].value;
			return;
		}
		else if (table[hash].key == 0) {
			// entry does not exist
			values[index] = 0;
			return;
		}
		hash = (hash + 1) % size;
		if (hash == start) {
			// we have looped through the entire table
			values[index] = 0;
			return;
		}
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// move keys to GPU
	int* keysDevice;
	glbGpuAllocator->_cudaMalloc((void**)&keysDevice, numKeys * sizeof(int));
	cudaMemcpy(keysDevice, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	// alloc return values
	int* values;
	glbGpuAllocator->_cudaMalloc((void**)&values, numKeys * sizeof(int));
	// call kernel
	getBatchKernel<<<(numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(table, capacity, keysDevice, values, numKeys);
	cudaDeviceSynchronize();
	// alloc values on host
	int* valuesHost = (int*)malloc(numKeys * sizeof(int));
	// move values to host
	cudaMemcpy(valuesHost, values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	// cleanup
	glbGpuAllocator->_cudaFree(values);
	glbGpuAllocator->_cudaFree(keysDevice);
	return valuesHost;
}

float GpuHashTable::loadFactor() {
	return (float)count / capacity;
}