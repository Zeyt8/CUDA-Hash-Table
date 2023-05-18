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

__device__ static unsigned int fnvHash(const char* str)
{
    const size_t length = sizeof(uint32_t);
    unsigned int hash = 2166136261u;
    for (size_t i = 0; i < length; ++i)
    {
        hash ^= *str++;
        hash *= 16777619u;
    }
    return hash;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) {
	glbGpuAllocator->_cudaMallocManaged((void**)&table, size * sizeof(HashTableItem*));
	GpuHashTable::size = size;
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
__global__ void reshapeKernel(HashTableItem* newTable, HashTableItem* table, int numBucketsReshape)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (table[index].key == 0) {
		return;
	}
	unsigned int hash = fnvHash((char*)&table[index].key) % numBucketsReshape;
	while (newTable[hash].key != 0) {
		hash = (hash + 1) % numBucketsReshape;
	}
	newTable[hash] = table[index];
}

void GpuHashTable::reshape(int numBucketsReshape) {
	HashTableItem* newTable;
	glbGpuAllocator->_cudaMallocManaged((void**)&newTable, numBucketsReshape * sizeof(HashTableItem*));
	reshapeKernel<<<numBucketsReshape / 256 + 1, 256>>>(newTable, table, numBucketsReshape);
	glbGpuAllocator->_cudaFree(table);
	table = newTable;
	size = numBucketsReshape;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
__global__ void insertBatchKernel(HashTableItem* table, int size, int* keys, int* values, int numKeys)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int hash = fnvHash((char*)&keys[index]) % size;
	while (table[hash].key != keys[index]) {
		hash = (hash + 1) % size;
	}
	HashTableItem item;
	item.key = keys[index];
	item.value = values[index];
	table[hash] = item;
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	if (count + numKeys > size * 0.8f) {
		reshape(size * 2);
	}
	insertBatchKernel<<<numKeys / 256 + 1, 256>>>(table, size, keys, values, numKeys);
	count += numKeys;
	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
__global__ void getBatchKernel(HashTableItem* table, int size, int* keys, int* values)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int hash = keys[index] % size;
	while (table[hash].key != keys[index]) {
		hash = (hash + 1) % size;
	}
	if (table[hash].key != 0) {
		values[index] = table[hash].value;
	} else {
		values[index] = 0;
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int* values;
	glbGpuAllocator->_cudaMallocManaged((void**)&values, numKeys * sizeof(int));
	getBatchKernel<<<numKeys / 256 + 1, 256>>>(table, size, keys, values);
	return values;
}

float GpuHashTable::loadFactor() {
	return (float)count / size;
}