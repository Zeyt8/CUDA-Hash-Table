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

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) {
	table = (HashTableItem**)glbGpuAllocator->_cudaMallocManaged(size * sizeof(HashTableItem*));
	for (int i = 0; i < size; i++) {
		table[i] = NULL;
	}
	GpuHashTable::size = size;
	count = 0;
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	for (int i = 0; i < size; i++) {
		if (table[i] != NULL) {
			glbGpuAllocator->_cudaFree(table[i]);
		}
	}
	glbGpuAllocator->_cudaFree(table);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
__global__ void reshapeKernel(HashTableItem** newTable, int numBucketsReshape)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (table[index] == NULL) {
		return;
	}
	unsigned int hash = fnvHash(table[index]->key) % numBucketsReshape;
	while (newTable[hash] != NULL) {
		hash = (hash + 1) % numBucketsReshape;
	}
	newTable[hash] = table[index];
}

void GpuHashTable::reshape(int numBucketsReshape) {
	HashTableItem** newTable = (HashTableItem**)glbGpuAllocator->_cudaMallocManaged(numBucketsReshape * sizeof(HashTableItem*));
	for (int i = 0; i < numBucketsReshape; i++) {
		newTable[i] = NULL;
	}
	reshapeKernel<<<numBucketsReshape / 256 + 1, 256>>>(newTable, numBucketsReshape);
	glbGpuAllocator->_cudaFree(table);
	table = newTable;
	size = numBucketsReshape;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
__global__ void insertBatchKernel(int* keys, int* values, int numKeys)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int hash = fnvHash(keys[index]) % size;
	while (table[hash] != NULL && table[hash]->key != keys[index]) {
		hash = (hash + 1) % size;
	}
	if (table[hash] != NULL) {
		table[hash]->value = values[index];
		return;
	}
	HashTableItem* item = (HashTableItem*)glbGpuAllocator->_cudaMallocManaged(sizeof(HashTableItem));
	item->key = keys[index];
	item->value = values[index];
	table[hash] = item;
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	if (count + numKeys > size * 0.8) {
		reshape(size * 2);
	}
	insertBatchKernel<<<numKeys / 256 + 1, 256>>>(keys, values, numKeys);
	count += numKeys;
	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
__global__ void getBatchKernel(int* keys, int* values)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int hash = keys[index] % size;
	while (table[hash] != NULL && table[hash]->key != keys[index]) {
		hash = (hash + 1) % size;
	}
	if (table[hash] != NULL) {
		values[index] = table[hash]->value;
	} else {
		values[index] = INT_MIN;
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int* values = (int*)glbGpuAllocator->cudaMallocManaged(numKeys * sizeof(int));
	getBatchKernel<<<numKeys / 256 + 1, 256>>>(keys, values);
	return values;
}

unsigned int GpuHashTable::fnvHash(const char* str)
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