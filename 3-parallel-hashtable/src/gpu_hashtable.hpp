#ifndef _HASHCPU_
#define _HASHCPU_

#include <stdint.h>

/**
 * Class GpuHashTable to implement functions
 */
struct HashTableItem
{
	uint32_t key = 0;
	uint32_t value = 0;
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
		unsigned int size;
		unsigned int count;
};

#endif
