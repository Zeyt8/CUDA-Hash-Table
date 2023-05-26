#ifndef _HASHCPU_
#define _HASHCPU_

#include <stdint.h>

#define LOAD_FACTOR 0.8f
#define BLOCK_SIZE 256

/**
 * Class GpuHashTable to implement functions
 */
struct HashTableItem
{
	int key = 0;
	int value = 0;
};

class GpuHashTable
{
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		float loadFactor();

		~GpuHashTable();

	private:
		HashTableItem *table;
		unsigned int size;	// capacity
		unsigned int count;	// number of elements
};

#endif
